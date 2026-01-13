package fr.vergne.ai.neuralnet;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.is;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;

import fr.vergne.ai.neuralnet.NeuralNet.MLP;
import fr.vergne.ai.neuralnet.NeuralNet.Operator;
import fr.vergne.ai.neuralnet.NeuralNet.ParameterNamer;
import fr.vergne.ai.neuralnet.NeuralNet.Value;

class NeuralNetTest {

	@Test
	void testBasicDataOperations() {
		Value value = Value.of(1).plus(2).minus(4).mult(-10).div(5);
		assertThat(value.data().get(), is(2.0));
	}

	@Test
	void testValueDataInit() {
		var x1 = new Value("x1", 2.0);
		var x2 = new Value("x2", 0.0);
		var w1 = new Value("w1", -3.0);
		var w2 = new Value("w2", 1.0);
		var bias = new Value("bias", 6.8813735870195432);
		var n = x1.mult(w1).plus(x2.mult(w2)).plus(bias).named("n");
		var o = n.calc(Operator.TANH);

		o.backward();

		assertThat(x1.data().get(), is(2.0));
		assertThat(x2.data().get(), is(0.0));
		assertThat(w1.data().get(), is(-3.0));
		assertThat(w2.data().get(), is(1.0));
		assertThat(bias.data().get(), is(6.8813735870195432));
		double epsilonCalc = 1e-10;
		assertThat(n.data().get(), is(closeTo(0.8813735870195432, epsilonCalc)));
		assertThat(o.data().get(), is(closeTo(0.7071067811865477, epsilonCalc)));
	}

	@Test
	void testValueGradientCompute() {
		var x1 = new Value("x1", 2.0);
		var x2 = new Value("x2", 0.0);
		var w1 = new Value("w1", -3.0);
		var w2 = new Value("w2", 1.0);
		var bias = new Value("bias", 6.8813735870195432);
		var n = x1.mult(w1).plus(x2.mult(w2)).plus(bias).named("n");
		var o = n.calc(Operator.TANH);

		o.backward();

		double epsilonGradient = 1e-10;
		assertThat(x1.gradient().get(), is(closeTo(-1.5, epsilonGradient)));
		assertThat(x2.gradient().get(), is(closeTo(0.5, epsilonGradient)));
		assertThat(w1.gradient().get(), is(closeTo(1.0, epsilonGradient)));
		assertThat(w2.gradient().get(), is(0.0));// No impact so not updated
		assertThat(bias.gradient().get(), is(closeTo(0.5, epsilonGradient)));
		assertThat(n.gradient().get(), is(closeTo(0.5, epsilonGradient)));
		assertThat(o.gradient().get(), is(1.0));// Root so set to 1.0
	}

	@Test
	void testValueReuse() {
		var a = Value.of(-2.0);
		var b = Value.of(3.0);
		var c = a.mult(b);// a used here
		var d = a.plus(b);// and here
		var e = c.mult(d);

		e.backward();

		assertThat(a.gradient().get(), is(-3.0));
		assertThat(b.gradient().get(), is(-8.0));
		assertThat(c.gradient().get(), is(1.0));
		assertThat(d.gradient().get(), is(-6.0));
		assertThat(e.gradient().get(), is(1.0));
	}

	@Test
	void testSimpleMlp() {
		Random random = new Random(0);
		MLP mlp = new MLP(ParameterNamer.create(), 3, List.of(4, 4, 1), (_) -> random.nextDouble(-1.0, 1.0));
		List<Value> x = Stream.of(2.0, 3.0, -1.0).map(Value::of).toList();
		Value result = mlp.compute(x).get(0);

		assertThat(result.data().get(), is(closeTo(-0.6836299751085826, 1e-10)));
	}

	@Test
	void testGradientDescent() {
		List<Double> x0 = List.of(2.0, 3.0, -1.0);
		double y0 = 1.0;
		List<Double> x1 = List.of(3.0, -1.0, 0.5);
		double y1 = -1.0;
		List<Double> x2 = List.of(0.5, 1.0, 1.0);
		double y2 = -1.0;
		List<Double> x3 = List.of(1.0, 1.0, -1.0);
		double y3 = 1.0;

		Map<List<Double>, Double> dataset = new LinkedHashMap<>();
		dataset.put(x0, y0);
		dataset.put(x1, y1);
		dataset.put(x2, y2);
		dataset.put(x3, y3);

		Random random = new Random(0);
		MLP mlp = new MLP(ParameterNamer.create(), 3, List.of(4, 4, 1), (_) -> random.nextDouble(-1.0, 1.0));
		Value loss = null;
		for (int i = 0; i < 30; i++) {
			loss = mlp.computeLoss(dataset);
			loss.backward();
			mlp.updateParameters(0.5);
		}

		double epsilonLoss = 1e-10;
		assertThat(loss.data().get(), is(closeTo(9.555022845608106e-5, epsilonLoss)));

		double epsilonOutput = 1e-2;
		assertThat(mlp.computeRaw(x0).get(0), is(closeTo(y0, epsilonOutput)));
		assertThat(mlp.computeRaw(x1).get(0), is(closeTo(y1, epsilonOutput)));
		assertThat(mlp.computeRaw(x2).get(0), is(closeTo(y2, epsilonOutput)));
		assertThat(mlp.computeRaw(x3).get(0), is(closeTo(y3, epsilonOutput)));

		// Only test the parameters of the first neuron, assuming the rest is as stable
		double epsilonParameter = 1e-10;
		assertThat(mlp.layer(0).neuron(0).weight(0).data().get(), is(closeTo(0.9680755614655998, epsilonParameter)));
		assertThat(mlp.layer(0).neuron(0).weight(1).data().get(), is(closeTo(0.2974987825173542, epsilonParameter)));
		assertThat(mlp.layer(0).neuron(0).weight(2).data().get(), is(closeTo(-0.5004395032020763, epsilonParameter)));
		assertThat(mlp.layer(0).neuron(0).bias().data().get(), is(closeTo(0.44813869830534536, epsilonParameter)));
	}
}
