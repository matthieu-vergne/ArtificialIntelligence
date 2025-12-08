package fr.vergne.ai.agent;

import fr.vergne.ai.context.Context;

/**
 * An {@link Agent} is the software embodiment of an artificial intelligence. It
 * can interact with its environment by registering {@link Sensor}s as inputs
 * and {@link Actuator}s as outputs. On can request
 * 
 * @author Matthieu Vergne <matthieu.vergne@gmail.com>
 *
 */
public interface Agent extends Runnable {

	/**
	 * Add a {@link Sensor} to this {@link Agent}. If one wants to change the
	 * {@link Sensor}, one needs first to remove it with
	 * {@link #removeSensor(Object)} before to add the new one. Redundant calls
	 * (assigning a second time the same {@link Sensor} to the same ID) are also
	 * rejected because no addition is performed.
	 * 
	 * @param id
	 *            the ID of the {@link Sensor} to add
	 * @param sensor
	 *            the {@link Sensor} to add
	 * @throws NullPointerException
	 *             if the ID or {@link Sensor} is <code>null</code>
	 * @throws IllegalArgumentException
	 *             if the ID is already used by another {@link Sensor}
	 */
	public void addSensor(Object id, Sensor sensor);

	/**
	 * Remove any {@link Sensor} associated to a given ID. If no {@link Sensor}
	 * is associated to this ID, nothing happen.
	 * 
	 * @param id
	 *            the ID of the {@link Sensor} to remove
	 */
	public void removeSensor(Object id);

	/**
	 * Provide the current {@link Sensor} associated to the given ID.
	 * 
	 * @param id
	 *            the ID of the {@link Sensor}
	 * @return the {@link Sensor} associated to the ID, <code>null</code> if
	 *         there is no such {@link Sensor}
	 */
	public Sensor getSensor(Object id);

	/**
	 * Add an {@link Actuator} to this {@link Agent}. If one wants to change the
	 * {@link Actuator}, one needs first to remove it with
	 * {@link #removeActuator(Object)} before to add the new one. Redundant
	 * calls (assigning a second time the same {@link Actuator} to the same ID)
	 * are also rejected because no addition is performed.
	 * 
	 * @param id
	 *            the ID of the {@link Actuator} to add
	 * @param actuator
	 *            the {@link Actuator} to add
	 * @throws NullPointerException
	 *             if the ID or {@link Actuator} is <code>null</code>
	 * @throws IllegalArgumentException
	 *             if the ID is already used by another {@link Actuator}
	 */
	public void addActuator(Object id, Actuator actuator);

	/**
	 * Remove any {@link Actuator} associated to a given ID. If no
	 * {@link Actuator} is associated to this ID, nothing happen.
	 * 
	 * @param id
	 *            the ID of the {@link Actuator} to remove
	 */
	public void removeActuator(Object id);

	/**
	 * Provide the current {@link Actuator} associated to the given ID.
	 * 
	 * @param id
	 *            the ID of the {@link Actuator}
	 * @return the {@link Actuator} associated to the ID, <code>null</code> if
	 *         there is no such {@link Actuator}
	 */
	public Actuator getActuator(Object id);

	/**
	 * 
	 * @param goal
	 *            {@link Context} the {@link Agent} should reach
	 */
	public void setGoal(Context goal);

	/**
	 * 
	 * @return the {@link Context} the {@link Agent} should reach
	 */
	public Context getGoal();

	/**
	 * Inform the agent whether the current state of the world corresponds to
	 * the {@link Context} provided as goal. The information provided is valid
	 * upon reception only: when this function is called, this is the current
	 * state of achievement, but it is not because nothing is sent later that
	 * this state did not evolve. A continuous confirmation should be sent to
	 * confirm that the state is still the same. The {@link Agent} may assume
	 * extra assumptions, but they are not part of the contract of this
	 * function.
	 * 
	 * TODO The current notion of achievement is boolean, not a scale, so there
	 * is no notion such as a degree of achievement or a distance towards the
	 * goal. It might be worth considering for future evolutions, but we might
	 * also expect the {@link Agent} itself to build internal evaluations
	 * allowing it to evaluate on its own the distance towards the goal
	 * depending on the remaining steps to do.
	 * 
	 * @param isAchieved
	 *            <code>true</code> if the current state of the world
	 *            corresponds to the goal, <code>false</code> otherwise.
	 */
	public void informGoalAchievement(boolean isAchieved);
}
