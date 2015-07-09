package fr.vergne.ai.impl;

import java.util.Collection;
import java.util.HashSet;

import fr.vergne.ai.Sensor;

public class ManualSensor<Value> implements Sensor<Value> {

	Collection<SensorListener<Value>> listeners = new HashSet<SensorListener<Value>>();

	@Override
	public void addSensorListener(SensorListener<Value> listener) {
		listeners.add(listener);
	}

	@Override
	public void removeSensorListener(SensorListener<Value> listener) {
		listeners.remove(listener);
	}

	public void push(Value value) {
		for (SensorListener<Value> listener : listeners) {
			listener.sensing(value);
		}
	}

}
