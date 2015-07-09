package fr.vergne.ai.impl;

import static org.junit.Assert.*;

import java.util.LinkedList;

import org.junit.Test;

import fr.vergne.ai.Sensor.SensorListener;

public class ManualSensorTest {

	@Test
	public void testPushProperlyNotifiesAddedListeners() {
		ManualSensor<Object> sensor = new ManualSensor<Object>();
		final LinkedList<Object> values = new LinkedList<Object>();
		sensor.addSensorListener(new SensorListener<Object>() {

			@Override
			public void sensing(Object value) {
				values.add(value);
			}
		});

		assertEquals(0, values.size());
		sensor.push(3);
		assertEquals(1, values.size());
		assertEquals(3, values.getLast());
		sensor.push("test");
		assertEquals(2, values.size());
		assertEquals("test", values.getLast());
		sensor.push(null);
		assertEquals(3, values.size());
		assertEquals(null, values.getLast());
	}

	@Test
	public void testPushDoesNotNotifyRemovedListeners() {
		ManualSensor<Object> sensor = new ManualSensor<Object>();
		final LinkedList<Object> values = new LinkedList<Object>();
		SensorListener<Object> listener = new SensorListener<Object>() {

			@Override
			public void sensing(Object value) {
				values.add(value);
			}
		};
		sensor.addSensorListener(listener);
		sensor.removeSensorListener(listener);

		assertEquals(0, values.size());
		sensor.push(3);
		assertEquals(0, values.size());
		sensor.push("test");
		assertEquals(0, values.size());
		sensor.push(null);
		assertEquals(0, values.size());
	}

}
