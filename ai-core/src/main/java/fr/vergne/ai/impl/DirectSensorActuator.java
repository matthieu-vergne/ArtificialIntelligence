package fr.vergne.ai.impl;

import fr.vergne.ai.Actuator;
import fr.vergne.ai.Sensor;

public class DirectSensorActuator<Value> implements Actuator, Sensor<Value> {

	private final ManualSensor<Value> sensor;
	private final ValueGenerator<Value> generator;

	public DirectSensorActuator(ValueGenerator<Value> generator) {
		this.sensor = new ManualSensor<Value>();
		this.generator = generator;
	}

	@Override
	public void act() {
		sensor.push(generator.generate());
	}

	public static interface ValueGenerator<Value> {
		public Value generate();
	}

	@Override
	public void addSensorListener(SensorListener<Value> listener) {
		sensor.addSensorListener(listener);
	}

	@Override
	public void removeSensorListener(SensorListener<Value> listener) {
		sensor.removeSensorListener(listener);
	}
}
