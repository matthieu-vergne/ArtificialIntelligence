package fr.vergne.ai.impl;

import static org.junit.Assert.*;

import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;

import org.junit.Test;

public class SensorDescriptorTest {

	@Test
	public void testSizeReturnsNumberOfObservedValues() {
		ManualSensor<String> sensor = new ManualSensor<String>();
		SensorDescriptor<String> descriptor = new SensorDescriptor<String>(
				sensor);

		sensor.push("a");
		sensor.push("b");
		sensor.push("c");
		sensor.push("c");

		assertEquals(3, descriptor.size());
	}

	@Test
	public void testIteratorReturnsObservedValues() {
		ManualSensor<String> sensor = new ManualSensor<String>();
		SensorDescriptor<String> descriptor = new SensorDescriptor<String>(
				sensor);

		sensor.push("a");
		sensor.push("b");
		sensor.push("c");
		sensor.push("c");

		Collection<String> values = new LinkedList<String>();
		for (String value : descriptor) {
			values.add(value);
		}
		assertEquals(3, values.size());
		assertTrue(values.contains("a"));
		assertTrue(values.contains("b"));
		assertTrue(values.contains("c"));
	}

	@Test
	public void testImportanceIncreasesWithRarity() {
		ManualSensor<String> sensor = new ManualSensor<String>();
		SensorDescriptor<String> descriptor = new SensorDescriptor<String>(
				sensor);

		sensor.push("a");
		sensor.push("b");
		sensor.push("a");
		sensor.push("a");

		double importanceA = descriptor.getImportance("a");
		double importanceB = descriptor.getImportance("b");
		assertTrue("A(" + importanceA + ") > B(" + importanceB + ")",
				importanceA < importanceB);
	}

	@Test
	public void testIteratorReturnsByImportance() {
		ManualSensor<String> sensor = new ManualSensor<String>();
		SensorDescriptor<String> descriptor = new SensorDescriptor<String>(
				sensor);

		Iterator<String> iterator;

		sensor.push("a");
		iterator = descriptor.iterator();
		assertEquals("a", iterator.next());

		sensor.push("b");
		sensor.push("a");
		iterator = descriptor.iterator();
		assertEquals("b", iterator.next());
		assertEquals("a", iterator.next());

		sensor.push("c");
		sensor.push("a");
		sensor.push("c");
		iterator = descriptor.iterator();
		assertEquals("b", iterator.next());
		assertEquals("c", iterator.next());
		assertEquals("a", iterator.next());
	}

	@Test
	public void testImportanceOfFirstIsOne() {
		ManualSensor<String> sensor = new ManualSensor<String>();
		SensorDescriptor<String> descriptor = new SensorDescriptor<String>(
				sensor);

		sensor.push("a");
		sensor.push("b");
		sensor.push("a");
		sensor.push("a");

		assertEquals(1.0,
				descriptor.getImportance(descriptor.iterator().next()), 0.0);
	}

}
