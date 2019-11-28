use super::audio::TrainingData;
use tensorflow::{
    ops, train::AdadeltaOptimizer, train::MinimizeOptions, train::Optimizer, DataType, Operation,
    Output, Scope, Session, SessionOptions, SessionRunArgs, Shape, Tensor, Variable,
};

// /// A model can classify fft samples.
// pub struct Model {
//     session: Session,
//     input_layer_placeholder: Operation,
//     output_layer: Output,
//     variables: Vec<Variable>,
// }

// impl Model {
//     pub fn create_initial(hidden_size: (u64, u64)) -> Self {
//         let input_height = super::common::FFT_BINS as u64;
//         let output_height = 2;
//         let (variables, mut scope, input, output) = model(input_height, hidden_size, output_height);

//         let session = Session::new(&SessionOptions::new(), &scope.graph()).unwrap();

//         // Create random initial values for variables
//         {
//             let mut initialize = SessionRunArgs::new();
//             for var in &variables {
//                 initialize.add_target(&var.initializer());
//             }
//             session.run(&mut initialize).unwrap();
//         }

//         Model {
//             session,
//             input_layer_placeholder: input,
//             output_layer: output,
//             variables,
//         }
//     }
// }

pub fn train(training_data: &TrainingData, passes: usize, hidden_size: (u64, u64)) {
    let input_height = super::common::FFT_BINS as u64;
    let output_height = 2;

    let (variables, mut scope, input, output) = model(input_height, hidden_size, output_height);
    let label = ops::Placeholder::new()
        .data_type(DataType::Float)
        .shape(Shape::from(&[1u64, output_height][..]))
        .build(&mut scope.with_op_name("label"))
        .unwrap();

    let error = ops::subtract(output.clone(), label.clone(), &mut scope).unwrap();
    let error_squared = ops::multiply(error.clone(), error, &mut scope).unwrap();
    let (minimizer_vars, minimize) = AdadeltaOptimizer::new()
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
    let g = scope.graph();
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
    let mut input_tensor = Tensor::<f32>::new(&[1, input_height]);
    let mut label_tensor = Tensor::<f32>::new(&[1, output_height]);

    for i in 0..passes {
        let count = (training_data.close_front_unrounded_vowel.len()
            + training_data.open_back_rounded_vowel.len()) as f32;
        let mut tote = 0.0f32;

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
            tote += run_args.fetch::<f32>(error_squared_fetch).unwrap()[0].powf(0.5);
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
            tote += run_args.fetch::<f32>(error_squared_fetch).unwrap()[0].powf(0.5);
        }

        if i % 100 == 0 || i == passes - 1 {
            // evaluate training progress
            let ave_err = tote / count;
            dbg!(ave_err);
        }
    }

    println!("done");
}

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
) -> (Vec<Variable>, Output) {
    let mut scope = scope.new_sub_scope("layer");
    let scope = &mut scope;
    let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope).unwrap();
    let w = Variable::builder()
        .initial_value(ops::random_normal(w_shape, scope).unwrap())
        .data_type(DataType::Float)
        .shape(Shape::from(&[input_size, output_size][..]))
        .build(&mut scope.with_op_name("w"))
        .unwrap();
    let b = Variable::builder()
        .const_initial_value(Tensor::<f32>::new(&[output_size]))
        .build(&mut scope.with_op_name("b"))
        .unwrap();
    (
        vec![w.clone(), b.clone()],
        activation(
            ops::add(
                ops::mat_mul(input, w.output().clone(), scope).unwrap(),
                b.output().clone(),
                scope,
            )
            .unwrap()
            .into(),
            scope,
        ),
    )
}

/// Generate an untrained nn for use in phoneme classification.
fn model(
    input_height: u64,
    hidden_size: (u64, u64),
    output_height: u64,
) -> (Vec<Variable>, Scope, Operation, Output) {
    let mut scope = Scope::new_root_scope();
    let mut variables = Vec::new();
    let (hidden_width, hidden_height) = hidden_size;

    let input = ops::Placeholder::new()
        .data_type(DataType::Float)
        .shape(Shape::from(&[1u64, input_height][..]))
        .build(&mut scope.with_op_name("input"))
        .unwrap();

    let mut latest_layer: Output = input.clone().into();
    let mut latest_height = input_height;

    // hidden layers
    for _ in 0..hidden_width {
        let (mut vars, next_layer) = layer(
            latest_layer,
            latest_height,
            hidden_height,
            &|x, scope| ops::tanh(x, scope).unwrap().into(),
            &mut scope,
        );
        variables.append(&mut vars);
        latest_layer = next_layer;
        latest_height = hidden_height;
    }

    let (mut vars, output) = layer(
        latest_layer,
        hidden_height,
        output_height,
        &|x, _| x,
        &mut scope,
    );
    variables.append(&mut vars);

    (variables, scope, input, output)
}
