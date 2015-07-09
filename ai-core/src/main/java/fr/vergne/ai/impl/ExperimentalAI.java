package fr.vergne.ai.impl;

import java.util.Collection;
import java.util.HashSet;

import fr.vergne.ai.Actuator;
import fr.vergne.ai.ArtificialIntelligence;
import fr.vergne.ai.Sensor;
import fr.vergne.ai.Sensor.SensorListener;

public class ExperimentalAI implements ArtificialIntelligence {

	private final Collection<SensorDescriptor<?>> sensorDescriptors = new HashSet<SensorDescriptor<?>>();
	private final Collection<Actuator> actuators = new HashSet<Actuator>();

	public <Value> void addSensor(final Sensor<Value> sensor) {
		SensorDescriptor<Value> descriptor = new SensorDescriptor<Value>(sensor);
		sensorDescriptors.add(descriptor);
		
		sensor.addSensorListener(new SensorListener<Value>() {

			@Override
			public void sensing(Value value) {
				// TODO exploit value
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
	}

	public void addActuator(Actuator actuator) {
		actuators.add(actuator);
	}

	public void run() {
		// TODO setup a forward neural network (sensors + diff. sensors -> act.
		// sensors)
		// TODO setup a backward neural network (act. sensors -> sensors + diff.
		// sensors)
		// TODO setup learning:
		// http://www-igm.univ-mlv.fr/~dr/XPOSE2002/Neurones/index.php?rubrique=Apprentissage
		// TODO main loop
	}

}
