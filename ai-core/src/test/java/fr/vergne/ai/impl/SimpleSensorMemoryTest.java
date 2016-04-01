package fr.vergne.ai.impl;

import static org.junit.Assert.*;

import org.junit.Test;

import fr.vergne.ai.SensorMemory;
import fr.vergne.ai.SensorMemoryTest;

public class SimpleSensorMemoryTest extends SensorMemoryTest<String> {

	@Override
	protected SensorMemory<String> generateSensorMemory() {
		SimpleSensorMemory<String> memory = new SimpleSensorMemory<String>(100);
		for (int i = 0; i < 10; i++) {
			memory.add("a");
		}
		for (int i = 0; i < 20; i++) {
			memory.add("b");
		}
		for (int i = 0; i < 30; i++) {
			memory.add("c");
		}
		for (int i = 0; i < 40; i++) {
			memory.add("d");
		}
		return memory;
	}

	@Test
	public void testMemoryProvidesCorrectCapacityAtInstanciation() {
		SimpleSensorMemory<String> memory = new SimpleSensorMemory<String>(1000);
		assertEquals(1000, memory.getCapacity());
	}

	@Test
	public void testMemoryProvidesCorrectCapacityUponUpdate() {
		SimpleSensorMemory<String> memory = new SimpleSensorMemory<String>(1000);
		memory.setCapacity(100);
		assertEquals(100, memory.getCapacity());
	}

	@Test
	public void testMemoryIncrementsWeightOfAddedValue() {
		SimpleSensorMemory<String> memory = new SimpleSensorMemory<String>(1000);
		assertEquals(0, memory.getWeight("a"));

		memory.add("a");
		assertEquals(1, memory.getWeight("a"));
		assertEquals(0, memory.getWeight("b"));

		memory.add("a");
		assertEquals(2, memory.getWeight("a"));
		assertEquals(0, memory.getWeight("b"));

		memory.add("b");
		assertEquals(2, memory.getWeight("a"));
		assertEquals(1, memory.getWeight("b"));

		memory.add("b");
		assertEquals(2, memory.getWeight("a"));
		assertEquals(2, memory.getWeight("b"));
	}

	@Test
	public void testMemoryCapacityIsNotExceeded() {
		SimpleSensorMemory<String> memory = new SimpleSensorMemory<String>(5);
		for (int i = 0; i < memory.getCapacity() * 2; i++) {
			memory.add("a");

			int sum = 0;
			for (String value : memory) {
				sum += memory.getWeight(value);
			}
			assertTrue(
					"Capacity exceeded: " + sum + " > " + memory.getCapacity(),
					memory.getCapacity() >= sum);
		}
	}

	@Test
	public void testMemoryCapacityIsFullyExploited() {
		SimpleSensorMemory<String> memory = new SimpleSensorMemory<String>(5);
		for (int i = 0; i < memory.getCapacity(); i++) {
			memory.add("a");
		}
		int sum = 0;
		for (String value : memory) {
			sum += memory.getWeight(value);
		}
		assertTrue("Capacity exceeded: " + sum + " > " + memory.getCapacity(),
				memory.getCapacity() <= sum);
	}

	@Test
	public void testMemoryDecrementsWeightOfOldValues() {
		SimpleSensorMemory<String> memory = new SimpleSensorMemory<String>(5);
		for (int i = 0; i < memory.getCapacity(); i++) {
			memory.add("a");
		}
		assertEquals(
				"Test corrupted: " + memory.getCapacity() + " ≠ "
						+ memory.getWeight("a"), memory.getCapacity(),
				memory.getWeight("a"));
		int previous = memory.getWeight("a");
		while (memory.getWeight("a") > 0) {
			memory.add("b");
			assertTrue(memory.getWeight("a") < previous);
			previous = memory.getWeight("a");
		}
	}

	@Test
	public void testFamiliarityProportionalToWeightWhenFull() {
		SimpleSensorMemory<String> memory = new SimpleSensorMemory<String>(100);
		for (int i = 0; i < 10; i++) {
			memory.add("a");
		}
		for (int i = 0; i < 20; i++) {
			memory.add("b");
		}
		for (int i = 0; i < 30; i++) {
			memory.add("c");
		}
		for (int i = 0; i < 40; i++) {
			memory.add("d");
		}
		assertEquals(0.1, memory.getFamiliarity("a"), 0.0);
		assertEquals(0.2, memory.getFamiliarity("b"), 0.0);
		assertEquals(0.3, memory.getFamiliarity("c"), 0.0);
		assertEquals(0.4, memory.getFamiliarity("d"), 0.0);
	}

	@Test
	public void testFamiliarityProportionalToWeightWhenNotFull() {
		SimpleSensorMemory<String> memory = new SimpleSensorMemory<String>(1000);
		for (int i = 0; i < 10; i++) {
			memory.add("a");
		}
		for (int i = 0; i < 20; i++) {
			memory.add("b");
		}
		for (int i = 0; i < 30; i++) {
			memory.add("c");
		}
		for (int i = 0; i < 40; i++) {
			memory.add("d");
		}
		assertEquals(0.1, memory.getFamiliarity("a"), 0.0);
		assertEquals(0.2, memory.getFamiliarity("b"), 0.0);
		assertEquals(0.3, memory.getFamiliarity("c"), 0.0);
		assertEquals(0.4, memory.getFamiliarity("d"), 0.0);
	}
}
