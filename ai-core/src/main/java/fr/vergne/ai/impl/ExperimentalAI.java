package fr.vergne.ai.impl;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import fr.vergne.ai.Actuator;
import fr.vergne.ai.ArtificialIntelligence;
import fr.vergne.ai.Sensor;
import fr.vergne.ai.Sensor.SensorListener;
import fr.vergne.ai.SensorMemory;

public class ExperimentalAI implements ArtificialIntelligence {

	private final Collection<SensorMemory<?>> sensorMemories = new HashSet<SensorMemory<?>>();
	private final Map<Sensor<?>, Object> objectives = new HashMap<Sensor<?>, Object>();
	private final Collection<Actuator> actuators = new HashSet<Actuator>();

	public <Value> void addSensor(final Sensor<Value> sensor) {
		final SensorMemory<Value> memory = new SimpleSensorMemory<Value>(1000);
		sensorMemories.add(memory);

		sensor.addSensorListener(new SensorListener<Value>() {

			@Override
			public void sensing(Value value) {
				memory.add(value);
			}
		});

		sensor.addSensorListener(new SensorListener<Value>() {

			@Override
			public void sensing(Value value) {
				if (value instanceof Number) {
					// TODO register a differential listener
					// sensor.addSensorListener(listener);
					// sensorListeners.add(listener);
				} else {
					// no differential to compute
				}
				sensor.removeSensorListener(this);
			}
		});

		objectives.put(sensor, null);
	}

	public void addActuator(Actuator actuator) {
		actuators.add(actuator);
	}

	@Override
	public <Value> void setObjective(Sensor<Value> sensor, Value objective) {
		if (objectives.containsKey(sensor)) {
			objectives.put(sensor, objective);
		} else {
			throw new IllegalArgumentException("Unkown sensor " + sensor
					+ ", add it first.");
		}
	}

	public void run() {
		// TODO setup a forward neural network (sensors + diff. sensors -> act.
		// sensors)
		// TODO setup a backward neural network (act. sensors -> sensors + diff.
		// sensors)
		// TODO setup learning:
		// http://www-igm.univ-mlv.fr/~dr/XPOSE2002/Neurones/index.php?rubrique=Apprentissage
		// TODO main loop (learn + satisfy objectives)
	}

}
