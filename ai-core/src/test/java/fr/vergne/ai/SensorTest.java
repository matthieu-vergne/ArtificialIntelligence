package fr.vergne.ai;

import static org.junit.Assert.*;

import java.util.LinkedList;

import org.junit.Test;

import fr.vergne.ai.Sensor.SensorListener;

public abstract class SensorTest<Value> {

	abstract protected Sensor<Value> generateSensor();

	abstract protected void waitSensing(Sensor<Value> sensor);

	@Test
	public void testSensesValuesWhenListening() {
		Sensor<Value> sensor = generateSensor();
		final LinkedList<Value> history = new LinkedList<Value>();
		sensor.addSensorListener(new SensorListener<Value>() {

			@Override
			public void sensing(Value value) {
				history.addLast(value);
			}
		});
		for (int i = 1; i <= 100; i++) {
			waitSensing(sensor);
			assertFalse("Nothing sensed after " + i + " waiting",
					history.isEmpty());
		}
	}

	@Test
	public void testDoesNotSenseValuesIfDiscardListening() {
		Sensor<Value> sensor = generateSensor();
		final LinkedList<Value> history = new LinkedList<Value>();
		SensorListener<Value> listener = new SensorListener<Value>() {

			@Override
			public void sensing(Value value) {
				history.addLast(value);
			}
		};
		sensor.addSensorListener(listener);
		sensor.removeSensorListener(listener);
		waitSensing(sensor);
		assertTrue("Still sensing after removing listener", history.isEmpty());
	}
}
