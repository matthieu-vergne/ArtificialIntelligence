package fr.vergne.ai.impl;

import fr.vergne.ai.ArtificialIntelligence;
import fr.vergne.ai.ArtificialIntelligenceTest;

public class ExperimentalAITest extends ArtificialIntelligenceTest {

	@Override
	protected ArtificialIntelligence generateAI() {
		return new ExperimentalAI();
	}

}
