package fr.vergne.ai.ttt;

import java.util.Date;

public class RunnableFactory {

	public Runnable createInfiniteLoop(Runnable runnable) {
		return new Runnable() {

			@Override
			public void run() {
				while (true) {
					runnable.run();
				}
			}
		};
	}

	public Runnable createDeadlineLoop(Runnable runnable, Date stop) {
		return new Runnable() {

			@Override
			public void run() {
				while (stop.after(new Date())) {
					runnable.run();
				}
			}
		};
	}
}
