package fr.vergne.ai;

import static org.junit.Assert.*;

import java.math.BigDecimal;
import java.util.Iterator;

import org.junit.Test;

public abstract class SensorMemoryTest<Value> {

	abstract protected SensorMemory<Value> generateSensorMemory();

	@Test
	public void testMemoryIsNotEmpty() {
		SensorMemory<Value> memory = generateSensorMemory();
		Iterator<Value> iterator = memory.iterator();
		assertNotNull("No iterator provided", iterator);
		assertTrue("Generic tests cannot be applied on an empty memory",
				iterator.hasNext());
	}

	@Test
	public void testMemorySizeCorrespondsToMemoryIterator() {
		SensorMemory<Value> memory = generateSensorMemory();
		Iterator<Value> iterator = memory.iterator();
		int sum = 0;
		while (iterator.hasNext()) {
			iterator.next();
			sum++;
		}
		assertEquals(sum, memory.size());
	}

	@Test
	public void testIteratorSortedByDecreasingFamiliarity() {
		SensorMemory<Value> memory = generateSensorMemory();
		Iterator<Value> iterator = memory.iterator();
		double previous = Double.MAX_VALUE;
		while (iterator.hasNext()) {
			Value value = iterator.next();
			double current = memory.getFamiliarity(value);
			assertTrue("Value " + value + " with familiarity " + current
					+ " higher than " + previous, previous >= current);

			previous = current;
		}
	}

	@Test
	public void testFamiliaritySumEqualsOne() {
		SensorMemory<Value> memory = generateSensorMemory();
		Iterator<Value> iterator = memory.iterator();
		BigDecimal sum = BigDecimal.ZERO;
		while (iterator.hasNext()) {
			Value value = iterator.next();
			double familiarity = memory.getFamiliarity(value);
			sum = sum.add(new BigDecimal(familiarity));
		}
		assertEquals(1.0, sum.doubleValue(), 1e-10);
	}

}
