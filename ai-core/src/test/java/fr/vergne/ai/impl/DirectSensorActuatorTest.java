package fr.vergne.ai.impl;

import static org.junit.Assert.*;

import org.junit.Test;

import fr.vergne.ai.Sensor;
import fr.vergne.ai.Sensor.SensorListener;
import fr.vergne.ai.SensorTest;
import fr.vergne.ai.impl.DirectSensorActuator.ValueGenerator;

public class DirectSensorActuatorTest extends SensorTest<Integer> {

	static class Incrementer implements ValueGenerator<Integer> {

		private int value = 0;

		@Override
		public Integer generate() {
			return value++;
		}

	}

	@Test
	public void testIncrementerProvidesProperValues() {
		Incrementer incrementer = new Incrementer();
		assertEquals(0, (int) incrementer.generate());
		assertEquals(1, (int) incrementer.generate());
		assertEquals(2, (int) incrementer.generate());
		assertEquals(3, (int) incrementer.generate());
		assertEquals(4, (int) incrementer.generate());
	}

	@Override
	protected Sensor<Integer> generateSensor() {
		return new DirectSensorActuator<Integer>(new Incrementer());
	}

	@Override
	protected void waitSensing(Sensor<Integer> sensor) {
		DirectSensorActuator<Integer> actuator = (DirectSensorActuator<Integer>) sensor;
		actuator.act();
	}

	@Test
	public void testActuatorSensesUponAction() {
		final int[] value = {0};
		final int[] sensed = {0};
		DirectSensorActuator<Integer> actuator = new DirectSensorActuator<Integer>(new ValueGenerator<Integer>() {

			@Override
			public Integer generate() {
				return value[0];
			}
		});
		actuator.addSensorListener(new SensorListener<Integer>() {
			
			@Override
			public void sensing(Integer value) {
				sensed[0] = value;
			}
		});
		
		value[0] = 5;
		assertFalse(value[0] == sensed[0]);
		actuator.act();
		assertEquals(value[0], sensed[0]);
		
		value[0] = 10;
		assertFalse(value[0] == sensed[0]);
		actuator.act();
		assertEquals(value[0], sensed[0]);
		
		value[0] = -3;
		assertFalse(value[0] == sensed[0]);
		actuator.act();
		assertEquals(value[0], sensed[0]);
	}
}
