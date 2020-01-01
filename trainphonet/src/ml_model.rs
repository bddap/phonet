use super::audio::TrainingData;
use super::common::{INPUT_HEIGHT, OUTPUT_HEIGHT};
use tensorflow::{
    ops, train::AdadeltaOptimizer, train::MinimizeOptions, train::Optimizer, DataType, Operation,
    Output, Scope, Session, SessionOptions, SessionRunArgs, Shape, Tensor, Variable,
};

/// A model can classify fft samples.
pub struct Model {
    session: Session,
    input_layer_placeholder: Operation,
    output_layer: Output,
    variables: Vec<Variable>,
    scope: Scope,
}

impl Model {
    pub fn create_initial(hidden_size: (u64, u64)) -> Self {
        let (variables, scope, input, output) = model(hidden_size);

        let session = Session::new(&SessionOptions::new(), &scope.graph()).unwrap();

        // Create random initial values for variables
        {
            let mut initialize = SessionRunArgs::new();
            for var in &variables {
                initialize.add_target(&var.initializer());
            }
            session.run(&mut initialize).unwrap();
        }

        Model {
            session,
            input_layer_placeholder: input,
            output_layer: output,
            variables,
            scope,
        }
    }

    pub fn train(&mut self, training_data: &TrainingData, passes: usize) {
        let label = ops::Placeholder::new()
            .data_type(DataType::Float)
            .shape(Shape::from(&[1u64, OUTPUT_HEIGHT][..]))
            .build(&mut self.scope.with_op_name("label"))
            .unwrap();

        let error =
            ops::subtract(self.output_layer.clone(), label.clone(), &mut self.scope).unwrap();
        let error_squared = ops::multiply(error.clone(), error, &mut self.scope).unwrap();
        let (minimizer_vars, minimize) = AdadeltaOptimizer::new()
            .minimize(
                &mut self.scope,
                error_squared.clone().into(),
                MinimizeOptions::default().with_variables(&self.variables),
            )
            .unwrap();

        // Initialize variables.
        let options = SessionOptions::new();
        let g = self.scope.graph();
        let session = Session::new(&options, &g).unwrap();
        let mut run_args = SessionRunArgs::new();
        // Initialize variables the optimizer defined.
        for var in &minimizer_vars {
            run_args.add_target(&var.initializer());
        }
        session.run(&mut run_args).unwrap();

        // Train the model.
        let training_samples = training_data.input_output_pairs();

        for i in 0..passes {
            let mut err_sum = 0.0f32;

            for (fft, classification) in &training_samples {
                let input_tensor = fft.to_input_tensor();
                let label_tensor = classification.to_output_tensor();
                let mut run_args = SessionRunArgs::new();
                run_args.add_target(&minimize);
                let error_squared_fetch = run_args.request_fetch(&error_squared, 0);
                run_args.add_feed(&self.input_layer_placeholder, 0, &input_tensor);
                run_args.add_feed(&label, 0, &label_tensor);
                session.run(&mut run_args).unwrap();
                err_sum += run_args.fetch::<f32>(error_squared_fetch).unwrap()[0].powf(0.5);
            }

            if i % 100 == 0 || i == passes - 1 {
                // evaluate training progress
                let ave_err = err_sum / training_samples.len() as f32;
                dbg!(ave_err);
            }
        }

        println!("done");
    }
}

// Helper for building a dense layer.
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
