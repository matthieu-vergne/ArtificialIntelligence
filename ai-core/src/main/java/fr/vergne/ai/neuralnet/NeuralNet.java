package fr.vergne.ai.neuralnet;

import static java.util.stream.Collectors.toMap;

import java.time.Duration;
import java.time.Instant;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class NeuralNet {

	public static void main(String[] args) {
		// Massive computation to stress performance & resilience
		Random random = new Random(0);
		List<Integer> layerSizes = new LinkedList<Integer>();
		// Create 100^100 neurons
		for (int i = 0; i < 100; i++) {
			layerSizes.add(100);
		}
		MLP mlp = new MLP(ParameterNamer.create(), 2, layerSizes, (_) -> random.nextDouble(-1.0, 1.0));
		Instant start = Instant.now();
		Value loss = mlp.computeLoss(Map.of(List.of(123.0, 321.0), 1.0));
		Instant end = Instant.now();
		System.out.println("Loss=" + loss + " in " + Duration.between(start, end));

		Instant start2 = Instant.now();
		loss.backward();
		Instant end2 = Instant.now();
		System.out.println("Backward in " + Duration.between(start2, end2));
	}

	public enum Operator {
		NONE(null, //
				(_) -> {
					throw new RuntimeException("Should not be called");
				}, //
				(operands) -> {
					restrict(operands, 0);// Nothing to resolve
					return Collections.emptyList();
				}//
		), //

		PLUS("+", //
				(operands) -> operands.stream().mapToDouble(d -> d).sum(), //
				(operands) -> {
					return operands.stream().map(operand -> new GradiatedValue(operand, 1.0)).toList();
				}//
		), //
		MINUS("-", //
				(operands) -> {
					restrict(operands, 2);
					return operands.get(0) - operands.get(1);
				}, //
				(operands) -> {
					return operands.stream().map(operand -> new GradiatedValue(operand, 1.0)).toList();
				}//
		), //
		MULT("*", //
				(operands) -> operands.stream().mapToDouble(d -> d).reduce((a, b) -> a * b).getAsDouble(), //
				(operands) -> {
					restrict(operands, 2);

					Value op1 = operands.get(0);
					Value op2 = operands.get(1);

					Double localGradient1 = op2.data().get();
					Double localGradient2 = op1.data().get();

					return List.of(new GradiatedValue(op1, localGradient1), new GradiatedValue(op2, localGradient2));
				}//
		), //
		POW("^", //
				(operands) -> {
					restrict(operands, 2);
					return Math.pow(operands.get(0), operands.get(1));
				}, //
				(operands) -> {
					restrict(operands, 2);

					Value op1 = operands.get(0);
					Value op2 = operands.get(1);

					Double v1 = op1.data().get();
					Double v2 = op2.data().get();

					double localGradient1 = v2 * Math.pow(v1, v2 - 1);

					return List.of(new GradiatedValue(op1, localGradient1));
				}//
		), //
		EXP("exp", //
				(operands) -> {
					restrict(operands, 1);
					return Math.exp(operands.get(0));
				}, //
				(operands) -> {
					restrict(operands, 1);

					Value op = operands.get(0);
					Double v = op.data().get();
					double localGradient = Math.exp(v);

					return List.of(new GradiatedValue(op, localGradient));
				}//
		), //
		TANH("tanh", //
				(operands) -> {
					restrict(operands, 1);
					return Math.tanh(operands.get(0));
				}, //
				(operands) -> {
					restrict(operands, 1);

					Value op = operands.get(0);
					double tanh = Math.tanh(op.data().get());
					double localGradient = 1 - tanh * tanh;

					return List.of(new GradiatedValue(op, localGradient));
				}//
		), //
		RELU("ReLU", // Rectified Linear Unit
				(operands) -> {
					restrict(operands, 1);
					Double v = operands.get(0);
					return v < 0 ? 0 : v;
				}, //
				(operands) -> {
					restrict(operands, 1);

					Value op = operands.get(0);
					Double v = op.data().get();
					double localGradient = v < 0 ? 0.0 : 1.0;

					return List.of(new GradiatedValue(op, localGradient));
				}//
		);

		private final String label;
		private final Function<List<Double>, Double> computer;
		private final Function<List<Value>, List<GradiatedValue>> gradientResolver;

		Operator(String label, Function<List<Double>, Double> computer,
				Function<List<Value>, List<GradiatedValue>> gradientResolver) {
			this.label = label;
			this.computer = computer;
			this.gradientResolver = gradientResolver;
		}

		public Double compute(List<Double> operands) {
			return computer.apply(operands);
		}

		public List<GradiatedValue> resolveGradients(List<Value> operands) {
			return gradientResolver.apply(operands);
		}

		@Override
		public String toString() {
			return label;
		}
	}

	public static record Value(String label, AtomicReference<Double> data, Operator operator, List<Value> operands,
			AtomicReference<Double> gradient) {

		@Override
		public final int hashCode() {
			// Optimize hashcode computation by replacing operands by their labels
			// It removes recursive computation over a whole network
			return Objects.hash(label, data, operator, operands.stream().map(Value::label).toList(), gradient);
		}

		public Value(String label, double data) {
			this(label, new AtomicReference<>(data), Operator.NONE, Collections.emptyList(), new AtomicReference<>());
		}

		public Value(String label, Function<String, Double> dataInitializer) {
			this(label, dataInitializer.apply(label));
		}

		public Value(Supplier<String> label, Function<String, Double> dataInitializer) {
			this(label.get(), dataInitializer);
		}

		public Value updateDataRecursivelyWeigthedWithGradients(double step) {
			if (operator == Operator.NONE) {
				data.accumulateAndGet(step * gradient.get(), (t, u) -> t + u);
			} else {
				operands.forEach(operand -> operand.updateDataRecursivelyWeigthedWithGradients(step));
				Double computed = operator
						.compute(operands.stream().map(Value::data).map(AtomicReference::get).toList());
				data.set(computed);
			}
			return this;
		}

		public Value named(String label) {
			return new Value(label, data, operator, operands, gradient);
		}

		public void setGradient(double gradient) {
			this.gradient().set(gradient);
		}

		private void addGradient(double gradientDelta) {
			this.gradient().accumulateAndGet(gradientDelta, (g1, g2) -> g1 + g2);
		}

		public Value plus(Value that) {
			return calc(Operator.PLUS, that);
		}

		public Value plus(double value) {
			return plus(Value.of(value));
		}

		public Value minus(Value that) {
			return calc(Operator.MINUS, that);
		}

		public Value minus(double value) {
			return minus(Value.of(value));
		}

		public Value mult(Value that) {
			return calc(Operator.MULT, that);
		}

		public Value mult(double value) {
			return mult(Value.of(value));
		}

		public Value div(Value that) {
			return mult(that.pow(-1));
		}

		public Value div(double value) {
			return div(Value.of(value));
		}

		public Value pow(Value that) {
			return calc(Operator.POW, that);
		}

		public Value pow(double value) {
			return pow(Value.of(value));
		}

		public Value exp() {
			return calc(Operator.EXP);
		}

		public Value calc(Operator operator) {
			Double computed = operator.compute(Stream.of(this).map(Value::data).map(AtomicReference::get).toList());
			AtomicReference<Double> data = new AtomicReference<>(computed);
			return new Value("" + operator + "(" + this.data().get() + ")", data, operator, List.of(this),
					new AtomicReference<>());
		}

		public Value calc(Operator operator, Value that) {
			Double computed = operator
					.compute(Stream.of(this, that).map(Value::data).map(AtomicReference::get).toList());
			AtomicReference<Double> data = new AtomicReference<>(computed);
			return new Value("" + this.data().get() + " " + operator + " " + that.data().get(), data, operator,
					List.of(this, that), new AtomicReference<>());
		}

		@Override
		public final String toString() {
			return label;
		}

		public static Value of(double value) {
			return new Value(Double.toString(value), value);
		}

		public void backward() {
			List<Value> topo = topologicalOrder();

			for (Value value : topo.reversed()) {
				value.setGradient(0.0);
			}

			this.addGradient(1.0);
			for (Value parentValue : topo.reversed()) {
				double parentGradient = parentValue.gradient().get();
				Operator parentOperator = parentValue.operator();
				List<Value> parentOperands = parentValue.operands();
				for (GradiatedValue gradiatedParentOperand : parentOperator.resolveGradients(parentOperands)) {
					Value parentOperand = gradiatedParentOperand.value();
					double localGradient = gradiatedParentOperand.gradient();
					double operandGradient = parentGradient * localGradient;
					parentOperand.addGradient(operandGradient);
				}
			}
		}

		private List<Value> topologicalOrder() {
			List<Value> topo = new LinkedList<>();
			Set<Value> visited = new HashSet<>();
			Deque<Value> queue = new LinkedList<>();
			queue.add(this);
			while (!queue.isEmpty()) {
				Value value = queue.removeFirst();
				if (visited.add(value)) {
					topo.addFirst(value);
					for (Value child : value.operands()) {
						queue.addLast(child);
					}
				}
			}

			return topo;
		}
	}

	public record Input(Object id) {
	}

	public record Neuron(List<Value> weights, Value bias) {
		public Neuron(ParameterNamer parameterNamer, int inputSize, Function<String, Double> paramInitializer) {
			this(IntStream.range(0, inputSize).mapToObj(i -> new Value(parameterNamer.atX(i), paramInitializer))
					.toList(), new Value(parameterNamer.atBias(), paramInitializer));
		}

		public Value weight(int index) {
			return weights.get(index);
		}

		public Value compute(List<Value> x) {
			Value dotProduct = IntStream.range(0, x.size())//
					.mapToObj(i -> weights.get(i).mult(x.get(i)))//
					.reduce(Value.of(0), Value::plus);
			Value activity = dotProduct.plus(bias);
			return activity.calc(Operator.TANH);
		}

		public List<Value> parameters() {
			return Stream.concat(weights.stream(), Stream.of(bias)).toList();
		}
	}

	public record Layer(List<Neuron> neurons) {
		public Layer(ParameterNamer parameterNamer, int inputSize, int neuronsCount,
				Function<String, Double> paramInitializer) {
			this(IntStream.range(0, neuronsCount)//
					.mapToObj(i -> new Neuron(parameterNamer.atNeuron(i), inputSize, paramInitializer))//
					.toList());
		}

		public Neuron neuron(int index) {
			return neurons.get(index);
		}

		public List<Value> compute(List<Value> x) {
			return neurons.stream().map(neuron -> neuron.compute(x)).toList();
		}

		public List<Value> parameters() {
			return neurons.stream().map(Neuron::parameters).flatMap(List::stream).toList();
		}
	}

	public record MLP(List<Layer> layers) {
		private MLP(ParameterNamer parameterNamer, List<Integer> layersInputSizes, List<Integer> layersNeuronsCounts,
				Function<String, Double> paramInitializer) {
			this(IntStream.range(0, layersNeuronsCounts.size())//
					.mapToObj(i -> new Layer(parameterNamer.atLayer(i), layersInputSizes.get(i),
							layersNeuronsCounts.get(i), paramInitializer))//
					.toList());
		}

		// TODO Provide separate outputs (like inputs)
		// TODO Parallelize computation
		public MLP(ParameterNamer parameterNamer, int inputSize, List<Integer> layersNeuronsCounts,
				Function<String, Double> paramInitializer) {
			this(parameterNamer,
					Stream.concat(Stream.of(inputSize),
							layersNeuronsCounts.subList(0, layersNeuronsCounts.size() - 1).stream()).toList(),
					layersNeuronsCounts, paramInitializer);
		}

		public Layer layer(int index) {
			return layers.get(index);
		}

		public List<Value> compute(List<Value> x) {
			return layers.stream().reduce(x, (vector, layer) -> layer.compute(vector), noCombiner());
		}

		public List<Double> computeRaw(List<Double> x) {
			List<Value> xValues = x.stream().map(Value::of).toList();
			List<Value> resultValues = compute(xValues);
			List<Double> resultRaw = resultValues.stream().map(Value::data).map(AtomicReference::get).toList();
			return resultRaw;
		}

		public List<Value> parameters() {
			return layers.stream().map(Layer::parameters).flatMap(List::stream).toList();
		}

		public Value computeLoss(Map<List<Double>, Double> dataset) {
			Map<Value, Double> results = dataset.entrySet().stream().collect(toMap(//
					datapoint -> this.compute(datapoint.getKey().stream().map(Value::of).toList()).get(0), //
					datapoint -> datapoint.getValue()//
			));
			Value loss = Value.of(0);

			// Sum individual losses
			for (Entry<Value, Double> entry : results.entrySet()) {
				Value prediction = entry.getKey();
				Double target = entry.getValue();
				Value localLoss = prediction.minus(target).pow(2);
				loss = loss.plus(localLoss);
			}

			// Normalize over dataset size
			// Keep comparable loss for batching, testing, etc.
			loss = loss.div(results.size());

			return loss;
		}

		public void updateParameters(double step) {
			for (Value parameter : this.parameters()) {
				parameter.data().accumulateAndGet(-step * parameter.gradient().get(), (a, b) -> a + b);
			}
		}

	}

	static double polynom(double x, Iterable<Double> factors) {
		Iterator<Double> iterator = factors.iterator();

		if (iterator.hasNext()) {
			double result = iterator.next();
			while (iterator.hasNext()) {
				result = result * x + iterator.next();
			}
			return result;
		} else {
			return 0;
		}
	}

	private static <T> void restrict(List<T> list, int expectedSize) {
		int actualSize = list.size();
		if (actualSize != expectedSize) {
			throw new IllegalArgumentException("Expect " + expectedSize + " items but got " + actualSize);
		}
	}

	private static <T> BinaryOperator<T> noCombiner() {
		return (_, _) -> {
			throw new RuntimeException("Not combination is expected");
		};
	}

	public interface ParameterNamer extends Supplier<String> {
		String get();

		public static ParameterNamer create() {
			return new ParameterNamer() {
				@Override
				public String get() {
					return "";
				}
			};
		}

		default public ParameterNamer append(String postfix) {
			return () -> get() + postfix;
		}

		default public ParameterNamer atLayer(int i) {
			return append("L" + i);
		}

		default public ParameterNamer atNeuron(int i) {
			return append("N" + i);
		}

		default public ParameterNamer atX(int i) {
			return append("X" + i);
		}

		default public ParameterNamer atBias() {
			return append("B");
		}
	}

	public interface NeuronBrowser {
		Object id();

		Stream<WeightedInputBrowser> weightedInputs();

		double weightWith(String inputId);

		double bias();

	}

	public interface WeightedInputBrowser extends InputBrowser {
		double weight();
	}

	public interface InputsBrowser {

		Object id();

		Stream<InputBrowser> inputs();
	}

	public interface ClusterBrowser {

		Object id();

		Stream<NeuronBrowser> neurons();
	}

	public interface InputBrowser {

		Object id();
	}

	public interface NeuralNetBrowser {
		InputsBrowser inputsCluster();

		Stream<ClusterBrowser> neuronsClusters();

		NeuronBrowser neuron(String neuronId);

		public static NeuralNetBrowser forMlp(MLP mlp, Supplier<String> inputIdSupplier,
				Supplier<String> neuronIdSupplier, Supplier<String> layerIdSupplier) {

			return new NeuralNetBrowser() {
				private final String inputsLayerId = layerIdSupplier.get();
				private final int inputsSize = mlp.layer(0).neuron(0).weights().size();
				private final List<String> inputIds = Stream.generate(inputIdSupplier).limit(inputsSize).toList();
				private final Map<Layer, ClusterBrowser> clusters = new LinkedHashMap<>();
				private final Function<Layer, ClusterBrowser> clusterInitializer = layer -> {
					String id = layerIdSupplier.get();
					return new ClusterBrowser() {

						@Override
						public Object id() {
							return id;
						}

						@Override
						public Stream<NeuronBrowser> neurons() {
							return indexRange(mlp, layer).stream().mapToObj(index -> "N" + index)
									.map(neuronId -> neuron(neuronId));
						}

					};
				};

				@Override
				public InputsBrowser inputsCluster() {
					return new InputsBrowser() {

						@Override
						public Object id() {
							return inputsLayerId;
						}

						@Override
						public Stream<InputBrowser> inputs() {
							return inputIds.stream().map(id -> {
								return new InputBrowser() {

									@Override
									public Object id() {
										return id;
									}

								};
							});
						}
					};
				}

				@Override
				public Stream<ClusterBrowser> neuronsClusters() {
					return mlp.layers().stream().map(layer -> clusters.computeIfAbsent(layer, clusterInitializer));
				}

				@Override
				public NeuronBrowser neuron(String neuronId) {
					if (!neuronId.matches("N[0-9]+")) {
						throw new IllegalArgumentException("Unrecognized neuron ID: " + neuronId);
					}
					int neuronIndex = Integer.parseInt(neuronId.substring(1));
					Neuron neuron = retrieveNeuron(neuronIndex);
					if (neuron == null) {
						throw new IllegalArgumentException("Neuron ID too large: " + neuronId);
					}
					Layer outLayer = retrieveLayer(neuronIndex);
					return createNeuronBrowser(mlp, neuronId, neuron, outLayer);
				}

				private IndexRange indexRange(MLP mlp, Layer layer) {
					int indexMin = mlp.layers().stream().takeWhile(previousLayer -> !previousLayer.equals(layer))
							.mapToInt(previousLayer -> previousLayer.neurons().size()).sum();
					int indexMax = indexMin + layer.neurons().size();
					IndexRange indexRange = new IndexRange(indexMin, indexMax);
					return indexRange;
				}

				private NeuronBrowser createNeuronBrowser(MLP mlp, String neuronId, Neuron neuron, Layer layer) {
					return new NeuronBrowser() {

						@Override
						public Object id() {
							return neuronId;
						}

						@Override
						public double bias() {
							return neuron.bias().data().get();
						}

						@Override
						public Stream<WeightedInputBrowser> weightedInputs() {
							Layer layerBefore = mlp.layers().stream()//
									.takeWhile(previousLayer -> !previousLayer.equals(layer))//
									.reduce((_, nextLayer) -> nextLayer)//
									.orElse(null);
							IndexRange indexRange;
							Function<Integer, String> idProvider;
							if (layerBefore != null) {
								indexRange = indexRange(mlp, layerBefore);
								idProvider = index -> "N" + index;
							} else {
								// inputs case
								int inputsCount = mlp.layer(0).neuron(0).weights().size();
								indexRange = new IndexRange(0, inputsCount);
								idProvider = index -> "I" + index;
							}
							return indexRange.stream().mapToObj(index -> new WeightedInputBrowser() {

								@Override
								public Object id() {
									return idProvider.apply(index);
								}

								@Override
								public double weight() {
									int indexInLayer = index - indexRange.min();
									return neuron.weights().get(indexInLayer).data().get();
								}
							});
						}

						@Override
						public double weightWith(String inputId) {
							if (inputId.matches("I[0-9]+") && !layer.equals(mlp.layer(0))) {
								throw new IllegalArgumentException(
										"Only neuron of first hidden layer can use input: " + inputId);
							} else if (!inputId.matches("[IN][0-9]+")) {
								throw new IllegalArgumentException("Unrecognized neuron ID: " + inputId);
							}
							String inputType = inputId.substring(0, 1);
							if (inputType.equals("I")) {
								int inputIndex = Integer.parseInt(inputId.substring(1));
								Double inputWeight = retrieveInputWeight(neuron, inputIndex);
								if (inputWeight == null) {
									throw new IllegalArgumentException("Input ID too large: " + neuronId);
								}
								return inputWeight;
							} else if (inputType.equals("N")) {
								int inputIndex = Integer.parseInt(inputId.substring(1));
								Layer inLayer = retrieveLayer(inputIndex);
								if (mlp.layers().indexOf(inLayer) != mlp.layers().indexOf(layer) - 1) {
									throw new IllegalArgumentException(inputId + " is not an input of " + neuronId);
								}
								Double weight = retrieveNeuronWeight(neuron, inputIndex);
								if (weight == null) {
									throw new IllegalArgumentException("Input ID too large: " + neuronId);
								}
								return weight;
							} else {
								throw new RuntimeException("Unsupported input type: " + inputType);
							}
						}

						private Double retrieveNeuronWeight(Neuron neuron, int inputIndex) {
							for (Layer layer : mlp.layers()) {
								List<Neuron> neurons = layer.neurons();
								int neuronsCount = neurons.size();
								if (inputIndex >= neuronsCount) {
									inputIndex -= neuronsCount;
								} else {
									return neuron.weight(inputIndex).data().get();
								}
							}
							return null;
						}
					};
				}

				private Double retrieveInputWeight(Neuron neuron, int inputIndex) {
					Value value = neuron.weight(inputIndex);
					if (value == null) {
						return null;
					} else {
						return value.data().get();
					}
				}

				private Neuron retrieveNeuron(int neuronIndex) {
					for (Layer layer : mlp.layers()) {
						List<Neuron> neurons = layer.neurons();
						int neuronsCount = neurons.size();
						if (neuronIndex >= neuronsCount) {
							neuronIndex -= neuronsCount;
						} else {
							return layer.neuron(neuronIndex);
						}
					}
					return null;
				}

				private Layer retrieveLayer(int neuronIndex) {
					for (Layer layer : mlp.layers()) {
						List<Neuron> neurons = layer.neurons();
						int neuronsCount = neurons.size();
						if (neuronIndex >= neuronsCount) {
							neuronIndex -= neuronsCount;
						} else {
							return layer;
						}
					}
					return null;
				}
			};
		}

	}

	record IndexRange(int min, int max) {
		public IndexRange {
			if (min > max) {
				throw new IllegalArgumentException("Min " + min + " > max " + max);
			}
		}

		IntStream stream() {
			return IntStream.range(min, max);
		}
	}

	record GradiatedValue(Value value, double gradient) {
	}
}
