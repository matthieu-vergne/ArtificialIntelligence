package fr.vergne.ai.neuralnet;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.BiFunction;

interface Dataset {
	public static Map<List<Double>, Double> moons(Random random) {
		BiFunction<Double, Double, Double> noise = (value, amplitude) -> {
			return value + random.nextDouble(-amplitude, amplitude);
		};
		Map<List<Double>, Double> dataset = new LinkedHashMap<>();
		for (double x = -1.0; x <= 1.0; x += 0.1) {
			double y1 = x * x - 0.7;
			double y2 = -y1;
			double dx = Math.PI / 10;
			dataset.put(List.of(noise.apply(x, 0.1) + dx, noise.apply(y1, 0.1)), 1.0);
			dataset.put(List.of(noise.apply(x, 0.1) - dx, noise.apply(y2, 0.1)), -1.0);
		}
		return dataset;
	}

	public static Map<List<Double>, Double> steepColumns(Random random) {
		Map<List<Double>, Double> dataset = new LinkedHashMap<>();
		for (int i = 0; i < 100; i++) {
			double x = random.nextDouble(-5, 5);
			double y = random.nextDouble(-5, 5);
			double value = Math.signum(y - Math.sin(10 * x) * 5);
			dataset.put(List.of(x, y), value);
		}
		return dataset;
	}

	public static Map<List<Double>, Double> columns2(Random random) {
		Map<List<Double>, Double> dataset = new LinkedHashMap<>();
		for (int i = 0; i < 100; i++) {
			double x = random.nextDouble(-5, 5);
			double y = random.nextDouble(-5, 5);
			double value = Math.signum(y - Math.sin(x) * 5);
			dataset.put(List.of(x, y), value);
		}
		return dataset;
	}

	public static Map<List<Double>, Double> columns1(Random random) {
		Map<List<Double>, Double> dataset = new LinkedHashMap<>();
		for (int i = 0; i < 100; i++) {
			double x = random.nextDouble(-5, 5);
			double y = random.nextDouble(-5, 5);
			double value = Math.signum(y - NeuralNet.polynom(x, List.of(-2.0, 5.0, 4.0, -3.0)));
			dataset.put(List.of(x, y), value);
		}
		return dataset;
	}

	public static Map<List<Double>, Double> circle(Random random) {
		Map<List<Double>, Double> dataset = new LinkedHashMap<>();
		for (int i = 0; i < 100; i++) {
			double x = random.nextDouble(-5, 5);
			double y = random.nextDouble(-5, 5);
			double value = Math.signum(Math.sqrt(x * x + y * y) - 3);
			dataset.put(List.of(x, y), value);
		}
		return dataset;
	}
}