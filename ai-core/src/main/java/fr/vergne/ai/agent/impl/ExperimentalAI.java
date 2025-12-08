package fr.vergne.ai.agent.impl;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import fr.vergne.ai.agent.Actuator;
import fr.vergne.ai.agent.Agent;
import fr.vergne.ai.agent.Sensor;
import fr.vergne.ai.context.Context;

public class ExperimentalAI implements Agent {

	private final Logger logger = Logger.getLogger(ExperimentalAI.class.getName());
	private final Map<Object, Sensor> sensors = new LinkedHashMap<Object, Sensor>();
	private final Map<Object, Actuator> actuators = new LinkedHashMap<Object, Actuator>();
	private Context goal;

	@Override
	public void addSensor(Object id, Sensor sensor) {
		if (id == null) {
			throw new NullPointerException("Null ID");
		} else if (sensor == null) {
			throw new NullPointerException("Null sensor");
		} else if (sensors.containsKey(id)) {
			throw new IllegalArgumentException("Already used ID: " + id);
		} else {
			sensors.put(id, sensor);
		}
	}

	@Override
	public Sensor getSensor(Object id) {
		return sensors.get(id);
	}

	@Override
	public void removeSensor(Object id) {
		sensors.remove(id);
	}

	@Override
	public void addActuator(Object id, Actuator actuator) {
		if (id == null) {
			throw new NullPointerException("Null ID");
		} else if (actuator == null) {
			throw new NullPointerException("Null actuator");
		} else if (actuators.containsKey(id)) {
			throw new IllegalArgumentException("Already used ID: " + id);
		} else {
			actuators.put(id, actuator);
		}
	}

	@Override
	public Actuator getActuator(Object id) {
		return actuators.get(id);
	}

	@Override
	public void removeActuator(Object id) {
		actuators.remove(id);
	}

	@Override
	public void setGoal(Context goal) {
		this.goal = goal;
	}
	
	@Override
	public Context getGoal() {
		return goal;
	}
	
	@Override
	public void informGoalAchievement(boolean isAchieved) {
		// TODO Auto-generated method stub
		
	}

	private long round = 0;

	public void run() {
		round++;
		String logPrefix = "[" + this + ":" + round + "] ";
		logger.log(Level.INFO, logPrefix + "New round: " + System.currentTimeMillis());

		Map<Object, Object> sensorVector = new LinkedHashMap<>();
		for (Entry<Object, Sensor> entry : sensors.entrySet()) {
			Object id = entry.getKey();
			Object value = entry.getValue().sense();
			sensorVector.put(id, value);
			logger.log(Level.INFO, logPrefix + id + "=" + value);
		}

		Random rand = new Random();
		if (rand.nextFloat() > 0.9) {
			logger.log(Level.INFO, logPrefix + "No action");
		} else {
			List<Object> actuatorKeys = new LinkedList<>(actuators.keySet());
			Collections.shuffle(actuatorKeys);
			Object id = actuatorKeys.get(0);
			logger.log(Level.INFO, logPrefix + id);
			actuators.get(id).act();
		}
	}
}
