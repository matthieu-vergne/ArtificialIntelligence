package fr.vergne.ai.impl;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeSet;

import fr.vergne.ai.Sensor;
import fr.vergne.ai.Sensor.SensorListener;

public class SensorDescriptor<Value> implements Iterable<Value> {

	private final Sensor<Value> sensor;
	private final Map<Value, BigInteger> valueCounter = new HashMap<Value, BigInteger>();
	private final TreeSet<Value> valueSorter = new TreeSet<Value>(
			new Comparator<Value>() {

				@Override
				public int compare(Value value1, Value value2) {
					if (value1 == null && value1 == value2 || value1 != null
							&& value1.equals(value2)) {
						return 0;
					} else {
						BigInteger count1;
						BigInteger count2;
						synchronized (SensorDescriptor.this) {
							count1 = valueCounter.get(value1);
							count2 = valueCounter.get(value2);
						}
						count1 = count1 == null ? BigInteger.ZERO : count1;
						count2 = count2 == null ? BigInteger.ZERO : count2;
						int comparison = count1.compareTo(count2);
						return comparison == 0 ? 1 : comparison;
					}
				}
			});
	private final SensorListener<Value> listener = new SensorListener<Value>() {

		@Override
		public void sensing(Value value) {
			/*
			 * FIXME How to decide the value of this step is not yet clear.
			 * Tests should be implemented to check that specific properties are
			 * ensured by the step.
			 */
			BigInteger step;
			if (size() == 0) {
				step = BigInteger.ONE;
			} else {
				BigInteger min;
				BigInteger max;
				synchronized (SensorDescriptor.this) {
					min = valueCounter.get(valueSorter.first());
					max = valueCounter.get(valueSorter.last());
				}
				BigInteger diff = max.subtract(min);
				step = diff.divide(BigInteger.valueOf(1000));
				step = step.max(BigInteger.ONE);
			}

			synchronized (SensorDescriptor.this) {
				valueSorter.remove(value);
				BigInteger count = valueCounter.get(value);
				if (count == null) {
					count = step;
				} else {
					count = count.add(step);
				}
				valueCounter.put(value, count);
				valueSorter.add(value);
				System.out.println(valueCounter);
				System.out.println(valueSorter);
			}
		}
	};

	public SensorDescriptor(Sensor<Value> sensor) {
		this.sensor = sensor;
		this.sensor.addSensorListener(listener);
	}

	public Sensor<Value> getSensor() {
		return sensor;
	}

	@Override
	protected void finalize() throws Throwable {
		super.finalize();
		sensor.removeSensorListener(listener);
	}

	public int size() {
		return valueSorter.size();
	}

	public double getImportance(Value value) {
		BigDecimal count;
		BigDecimal countRef;
		synchronized (this) {
			count = new BigDecimal(valueCounter.get(value));
			Value ref = valueSorter.first();
			countRef = new BigDecimal(valueCounter.get(ref));
		}
		BigDecimal ratio = countRef.divide(count, 20, RoundingMode.HALF_UP);
		return ratio.doubleValue();
	}

	@Override
	public Iterator<Value> iterator() {
		return valueSorter.iterator();
	}
}
