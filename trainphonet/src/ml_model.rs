use super::audio::TrainingData;
use std::error::Error;
use tensorflow::{
    ops, train::AdadeltaOptimizer, train::MinimizeOptions, train::Optimizer, DataType, Output,
    Scope, Session, SessionOptions, SessionRunArgs, Shape, Status, Tensor, Variable,
};

// Helper for building a layer.
//
// `activation` is a function which takes a tensor and applies an activation
// function such as tanh.
//
// Returns variables created and the layer output.
fn layer<O1: Into<Output>>(
    input: O1,
    input_size: u64,
    output_size: u64,
    activation: &dyn Fn(Output, &mut Scope) -> Output,
    scope: &mut Scope,
) -> Result<(Vec<Variable>, Output), Status> {
    let mut scope = scope.new_sub_scope("layer");
    let scope = &mut scope;
    let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;
    let w = Variable::builder()
        .initial_value(ops::random_normal(w_shape, scope)?)
        .data_type(DataType::Float)
        .shape(Shape::from(&[input_size, output_size][..]))
        .build(&mut scope.with_op_name("w"))?;
    let b = Variable::builder()
        .const_initial_value(Tensor::<f32>::new(&[output_size]))
        .build(&mut scope.with_op_name("b"))?;
    Ok((
        vec![w.clone(), b.clone()],
        activation(
            ops::add(
                ops::mat_mul(input, w.output().clone(), scope)?,
                b.output().clone(),
                scope,
            )?
            .into(),
            scope,
        ),
    ))
}

pub fn train(training_data: &TrainingData) {
    let input_size = super::common::FFT_BINS as u64;
    let hidden_size = super::common::FFT_BINS as u64;
    let output_size = 2;

    let mut scope = Scope::new_root_scope();
    let input = ops::Placeholder::new()
        .data_type(DataType::Float)
        .shape(Shape::from(&[1u64, input_size][..]))
        .build(&mut scope.with_op_name("input"))
        .unwrap();
    let label = ops::Placeholder::new()
        .data_type(DataType::Float)
        .shape(Shape::from(&[1u64, output_size][..]))
        .build(&mut scope.with_op_name("label"))
        .unwrap();

    // Hidden layer.
    let (vars1, layer1) = layer(
        input.clone(),
        input_size,
        hidden_size,
        &|x, scope| ops::tanh(x, scope).unwrap().into(),
        &mut scope,
    )
    .unwrap();

    let (vars2, output) = layer(
        layer1.clone(),
        hidden_size,
        output_size,
        &|x, _| x,
        &mut scope,
    )
    .unwrap();

    let error = ops::subtract(output.clone(), label.clone(), &mut scope).unwrap();
    let error_squared = ops::multiply(error.clone(), error, &mut scope).unwrap();
    let mut optimizer = AdadeltaOptimizer::new();
    optimizer.set_learning_rate(ops::constant(1.0f32, &mut scope).unwrap());
    let mut variables = Vec::new();
    variables.extend(vars1);
    variables.extend(vars2);
    let (minimizer_vars, minimize) = optimizer
        .minimize(
            &mut scope,
            error_squared.clone().into(),
            MinimizeOptions::default().with_variables(&variables),
        )
        .unwrap();

    // =========================
    // Initialize the variables.
    // =========================
    let options = SessionOptions::new();
    let g = scope.graph_mut();
    let session = Session::new(&options, &g).unwrap();
    let mut run_args = SessionRunArgs::new();
    // Initialize variables we defined.
    for var in &variables {
        run_args.add_target(&var.initializer());
    }
    // Initialize variables the optimizer defined.
    for var in &minimizer_vars {
        run_args.add_target(&var.initializer());
    }
    session.run(&mut run_args).unwrap();

    // ================
    // Train the model.
    // ================
    let mut input_tensor = Tensor::<f32>::new(&[1, input_size]);
    let mut label_tensor = Tensor::<f32>::new(&[1, output_size]);

    // label for "i"
    label_tensor[0] = 1.0;
    label_tensor[1] = 0.0;
    for fft in &training_data.close_front_unrounded_vowel {
        input_tensor.copy_from_slice(&fft.0);
        let mut run_args = SessionRunArgs::new();
        run_args.add_target(&minimize);
        let error_squared_fetch = run_args.request_fetch(&error_squared, 0);
        run_args.add_feed(&input, 0, &input_tensor);
        run_args.add_feed(&label, 0, &label_tensor);
        session.run(&mut run_args).unwrap();
    }

    // label for "ɒ"
    label_tensor[0] = 0.0;
    label_tensor[1] = 1.0;
    for fft in &training_data.open_back_rounded_vowel {
        input_tensor.copy_from_slice(&fft.0);
        let mut run_args = SessionRunArgs::new();
        run_args.add_target(&minimize);
        let error_squared_fetch = run_args.request_fetch(&error_squared, 0);
        run_args.add_feed(&input, 0, &input_tensor);
        run_args.add_feed(&label, 0, &label_tensor);
        session.run(&mut run_args).unwrap();
    }

    {
        input_tensor.copy_from_slice(
            &training_data.open_back_rounded_vowel[training_data.open_back_rounded_vowel.len() / 2]
                .0,
        );

        // let mut args = SessionRunArgs::new();
        // args.add_feed(&input, 0, &input_tensor);
        // args.add_feed(&label, 0, &label_tensor);
        // let result_token = args.request_fetch(&label, 0);
        // session.run(&mut args).unwrap();
        // // let result_tensor = args.fetch::<f32>(result_token).unwrap();
    }

    println!("done");
}
