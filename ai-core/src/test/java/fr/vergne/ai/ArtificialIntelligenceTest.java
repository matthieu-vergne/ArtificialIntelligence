package fr.vergne.ai;

import static org.junit.Assert.*;

import org.junit.Test;

import fr.vergne.ai.impl.ManualSensor;

public abstract class ArtificialIntelligenceTest {

	abstract protected ArtificialIntelligence generateAI();

	@Test
	public void testObjectiveThrowsExceptionForUnknownSensor() {
		ArtificialIntelligence ai = generateAI();
		try {
			ai.setObjective(new ManualSensor<Integer>(), 10);
			fail("No exception thrown");
		} catch (IllegalArgumentException e) {
		}
	}

}
