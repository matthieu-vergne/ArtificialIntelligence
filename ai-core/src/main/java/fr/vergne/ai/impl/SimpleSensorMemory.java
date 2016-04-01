package fr.vergne.ai.impl;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import fr.vergne.ai.SensorMemory;

public class SimpleSensorMemory<Value> implements SensorMemory<Value> {

	private final LinkedList<Value> history = new LinkedList<Value>();
	private final Map<Value, Tuple<Value>> valueMap = new HashMap<Value, Tuple<Value>>();
	private int capacity;

	public SimpleSensorMemory(int capacity) {
		this.capacity = capacity;
	}

	public int getCapacity() {
		return capacity;
	}

	public void setCapacity(int capacity) {
		this.capacity = capacity;
	}

	@Override
	public synchronized void add(Value value) {
		while (history.size() >= capacity) {
			Value removed = history.removeFirst();
			Tuple<Value> tuple = valueMap.get(removed);
			if (tuple.weight == 1) {
				valueMap.remove(removed);
			} else {
				tuple.weight--;
			}
		}

		history.addLast(value);
		Tuple<Value> tuple = valueMap.get(value);
		if (tuple == null) {
			tuple = new Tuple<Value>(value, 1);
			valueMap.put(value, tuple);
		} else {
			tuple.weight++;
		}
	}

	@Override
	public synchronized double getFamiliarity(Value value) {
		return (double) getWeight(value) / history.size();
	}

	public synchronized int getWeight(Value value) {
		Tuple<Value> tuple = valueMap.get(value);
		if (tuple == null) {
			return 0;
		} else {
			return tuple.weight;
		}
	}

	@Override
	public Iterator<Value> iterator() {
		return new Iterator<Value>() {

			private final List<Tuple<Value>> remaining = new LinkedList<Tuple<Value>>(
					valueMap.values());

			@Override
			public boolean hasNext() {
				return !remaining.isEmpty();
			}

			@Override
			public Value next() {
				synchronized (SimpleSensorMemory.this) {
					// TODO optimize by using a map with weights as keys
					Collections.sort(remaining, familiarityComparator);
					return remaining.remove(0).value;
				}
			}

			@Override
			public void remove() {
				throw new RuntimeException(
						"You cannot alter the sensor memory through its iterator.");
			}
		};
	}

	@Override
	public int size() {
		return valueMap.size();
	}

	static class Tuple<Value> {
		private int weight;
		private final Value value;

		public Tuple(Value value, int weight) {
			this.value = value;
			this.weight = weight;
		}
	}

	private final Comparator<? super Tuple<Value>> familiarityComparator = new Comparator<Tuple<Value>>() {

		@Override
		public int compare(Tuple<Value> t1, Tuple<Value> t2) {
			return Integer.valueOf(t2.weight).compareTo(t1.weight);
		}
	};
}
