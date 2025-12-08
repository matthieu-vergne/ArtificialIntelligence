package fr.vergne.ai.impl;

import static org.junit.jupiter.api.Assertions.fail;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import fr.vergne.ai.AgentTest;
import fr.vergne.ai.agent.Agent;
import fr.vergne.ai.agent.impl.ExperimentalAI;

public class ExperimentalAITest implements AgentTest {

	@Override
	public Agent createAgent() {
		return new ExperimentalAI();
	}

	@Test
	@Disabled
	public void testXxx() {
		fail("Not yet implemented");
	}

}
