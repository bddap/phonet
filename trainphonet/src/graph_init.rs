use super::common::{INPUT_HEIGHT, OUTPUT_HEIGHT};
use tensorflow::{
    ops::{self, Placeholder},
    train::AdadeltaOptimizer,
    train::MinimizeOptions,
    train::Optimizer,
    DataType, Graph, Operation, Output, Scope, Session, SessionOptions, SessionRunArgs, Shape,
    Tensor, Variable,
};

/// Generate the model used for classifying vowels.
pub fn create_graph(hidden_size: (u64, u64)) -> Scope {
    let (hidden_width, hidden_height) = hidden_size;
    let mut scope = Scope::new_root_scope();
    let mut variables: Vec<Variable> = Vec::new();

    let input = ops::Placeholder::new()
        .data_type(DataType::Float)
        .shape(Shape::from(&[1u64, INPUT_HEIGHT][..]))
        .build(&mut scope.with_op_name("input"))
        .unwrap();

    let mut latest_layer: Output = input.clone().into();
    let mut latest_height = INPUT_HEIGHT;

    // dense hidden layers
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

    // This output layer represents the classified vowel.
    let (mut vars, output) = layer(
        latest_layer,
        hidden_height,
        OUTPUT_HEIGHT,
        &|x, _| x,
        &mut scope,
    );
    variables.append(&mut vars);

    // Training output gets loaded into this layer, this layer is compared to the output layer
    // during training.
    let label = ops::Placeholder::new()
        .data_type(DataType::Float)
        .shape(Shape::from(&[1u64, OUTPUT_HEIGHT][..]))
        .build(&mut scope.with_op_name("label"))
        .unwrap();

    // error symbol
    let error = ops::subtract(output.clone(), label.clone(), &mut scope).unwrap();
    let error_squared = ops::multiply(error.clone(), error, &mut scope).unwrap();
    let (minimizer_vars, minimize) = AdadeltaOptimizer::new()
        .minimize(
            &mut scope,
            error_squared.clone().into(),
            MinimizeOptions::default().with_variables(&variables),
        )
        .unwrap();

	scope.graph().deref().
	
    //(variables, scope, input, output)
    scope
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
