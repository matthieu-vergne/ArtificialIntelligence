package fr.vergne.ai;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import fr.vergne.ai.agent.Actuator;
import fr.vergne.ai.agent.Agent;
import fr.vergne.ai.agent.Sensor;

public interface AgentTest {

	Agent createAgent();

	@Test
	default void testGetSensorRetrievesAddedSensor() {
		Agent agent = createAgent();

		Object id1 = "ID1";
		Sensor sensor1 = () -> 1;
		agent.addSensor(id1, sensor1);

		Object id2 = "ID2";
		Sensor sensor2 = () -> 2;
		agent.addSensor(id2, sensor2);

		Object id3 = "ID3";
		Sensor sensor3 = () -> 3;
		agent.addSensor(id3, sensor3);

		// Check test validity
		assertNotEquals(id1, id2);
		assertNotEquals(id1, id3);
		assertNotEquals(id2, id3);
		assertNotEquals(sensor1, sensor2);
		assertNotEquals(sensor1, sensor3);
		assertNotEquals(sensor2, sensor3);

		// Test
		assertEquals(sensor1, agent.getSensor(id1));
		assertEquals(sensor2, agent.getSensor(id2));
		assertEquals(sensor3, agent.getSensor(id3));
	}

	@Test
	default void testGetSensorRetrievesNullForInexistantSensor() {
		Agent agent = createAgent();
		assertNull(agent.getSensor("ID"));
	}

	@Test
	default void testAddSensorRejectsNullID() {
		Agent agent = createAgent();
		Sensor sensor = () -> 1;

		try {
			agent.addSensor(null, sensor);
			fail("No exception thrown");
		} catch (NullPointerException e) {
		}
	}

	@Test
	default void testAddSensorRejectsNullSensor() {
		Agent agent = createAgent();
		Object id = "ID";

		try {
			agent.addSensor(id, null);
			fail("No exception thrown");
		} catch (NullPointerException e) {
		}
	}

	@Test
	default void testAddSensorRejectsRedundantCall() {
		Agent agent = createAgent();
		Object id = "ID";
		Sensor sensor1 = () -> 1;
		agent.addSensor(id, sensor1);

		// Check test validity
		assertNotNull(agent.getSensor(id));

		// Test
		try {
			agent.addSensor(id, sensor1);
			fail("No exception thrown");
		} catch (IllegalArgumentException e) {
		}
	}

	@Test
	default void testAddSensorRejectsAlreadyUsedID() {
		Agent agent = createAgent();
		Object id = "ID";
		Sensor sensor1 = () -> 1;
		agent.addSensor(id, sensor1);

		// Check test validity
		assertNotNull(agent.getSensor(id));

		// Test
		Sensor sensor2 = () -> 2;
		try {
			agent.addSensor(id, sensor2);
			fail("No exception thrown");
		} catch (IllegalArgumentException e) {
		}
	}

	@Test
	default void testRemoveSensorDeletesAddedSensor() {
		Agent agent = createAgent();
		Object id = "ID";
		Sensor sensor1 = () -> 1;
		agent.addSensor(id, sensor1);

		// Check test validity
		assertNotNull(agent.getSensor(id));

		// Test
		agent.removeSensor(id);
		assertNull(agent.getSensor(id));
	}

	@Test
	default void testRemoveSensorDoesNotFailOnAbsentSensor() {
		Agent agent = createAgent();
		Object id = "ID";

		// Check test validity
		assertNull(agent.getSensor(id));

		// Test
		agent.removeSensor(id);
	}

	@Test
	default void testGetActuatorRetrievesAddedActuator() {
		Agent agent = createAgent();

		Object id1 = "ID1";
		Actuator actuator1 = () -> {};
		agent.addActuator(id1, actuator1);

		Object id2 = "ID2";
		Actuator actuator2 = () -> {};
		agent.addActuator(id2, actuator2);

		Object id3 = "ID3";
		Actuator actuator3 = () -> {};
		agent.addActuator(id3, actuator3);

		// Check test validity
		assertNotEquals(id1, id2);
		assertNotEquals(id1, id3);
		assertNotEquals(id2, id3);
		assertNotEquals(actuator1, actuator2);
		assertNotEquals(actuator1, actuator3);
		assertNotEquals(actuator2, actuator3);

		// Test
		assertEquals(actuator1, agent.getActuator(id1));
		assertEquals(actuator2, agent.getActuator(id2));
		assertEquals(actuator3, agent.getActuator(id3));
	}

	@Test
	default void testGetActuatorRetrievesNullForInexistantActuator() {
		Agent agent = createAgent();
		assertNull(agent.getActuator("ID"));
	}

	@Test
	default void testAddActuatorRejectsNullID() {
		Agent agent = createAgent();
		Actuator actuator = () -> {};

		try {
			agent.addActuator(null, actuator);
			fail("No exception thrown");
		} catch (NullPointerException e) {
		}
	}

	@Test
	default void testAddActuatorRejectsNullActuator() {
		Agent agent = createAgent();
		Object id = "ID";

		try {
			agent.addActuator(id, null);
			fail("No exception thrown");
		} catch (NullPointerException e) {
		}
	}

	@Test
	default void testAddActuatorRejectsRedundantCall() {
		Agent agent = createAgent();
		Object id = "ID";
		Actuator actuator1 = () -> {};
		agent.addActuator(id, actuator1);

		// Check test validity
		assertNotNull(agent.getActuator(id));

		// Test
		try {
			agent.addActuator(id, actuator1);
			fail("No exception thrown");
		} catch (IllegalArgumentException e) {
		}
	}

	@Test
	default void testAddActuatorRejectsAlreadyUsedID() {
		Agent agent = createAgent();
		Object id = "ID";
		Actuator actuator1 = () -> {};
		agent.addActuator(id, actuator1);

		// Check test validity
		assertNotNull(agent.getActuator(id));

		// Test
		Actuator actuator2 = () -> {};
		try {
			agent.addActuator(id, actuator2);
			fail("No exception thrown");
		} catch (IllegalArgumentException e) {
		}
	}

	@Test
	default void testRemoveActuatorDeletesAddedActuator() {
		Agent agent = createAgent();
		Object id = "ID";
		Actuator actuator1 = () -> {};
		agent.addActuator(id, actuator1);

		// Check test validity
		assertNotNull(agent.getActuator(id));

		// Test
		agent.removeActuator(id);
		assertNull(agent.getActuator(id));
	}

	@Test
	default void testRemoveActuatorDoesNotFailOnAbsentActuator() {
		Agent agent = createAgent();
		Object id = "ID";

		// Check test validity
		assertNull(agent.getActuator(id));

		// Test
		agent.removeActuator(id);
	}

	@Disabled
	@Test
	default void testGoalMethods() {
		// TODO
		fail("Not implemented yet");
	}
}
