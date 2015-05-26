package br.usp.icmc.lapad;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import br.usp.icmc.lapad.ireos.IREOS;
import br.usp.icmc.lapad.ireos.IREOSSolution;
import br.usp.icmc.lapad.ireos.IREOSStatistics;

import com.rapidminer.operator.learner.functions.kernel.jmysvm.examples.SVMExamples;

public class Main {
	private static final String DATA = "WBC_withoutdupl_norm";
	private static final String LABELS = "solutions";

	public static void main(String[] args) throws Exception {
		List<int[]> ireosSolutions = new ArrayList<>();

		/* List all the solutions in the folder */
		File[] solutions = (new File(LABELS)).listFiles();

		/* Count the number of observations in the dataset */
		BufferedReader reader = new BufferedReader(new FileReader(solutions[0]));
		int datasetLength = 0;
		while (reader.readLine() != null) {
			datasetLength++;
		}

		/* Read all the solutions from the files and add in the list of vector */
		int outlierID = 0;
		for (int i = 0; i < solutions.length; i++) {
			String line = null;
			reader = new BufferedReader(new FileReader(solutions[i]));

			int detection[] = new int[datasetLength];
			outlierID = 0;
			while ((line = reader.readLine()) != null) {
				if (line.equals("\"outlier\"")) {
					detection[outlierID] = 1;
				} else {
					detection[outlierID] = -1;
				}
				outlierID++;
			}
			ireosSolutions.add(detection);
		}
		reader.close();

		/* Create the dataset model */
		SVMExamples examples = new SVMExamples(new BufferedReader(
				new FileReader(DATA)), datasetLength, 1000);

		/* Initialize IREOS using the dataset and the solutions to be evaluated */
		IREOS ireos = new IREOS(examples, ireosSolutions);

		/* Find the gamma maximum */
		ireos.findGammaMax();
		/* Set the number of values that gamma will be discretized */
		ireos.setnGamma(100);
		/*
		 * Discretize gamma from 0 to gammaMax into nGamma values, 3 for legacy
		 * logarithmic discretization
		 */
		ireos.setGammas(3);
		/* Set the maximum clump size */
		ireos.setmCl(1);
		/* Evaluate the solutions */
		List<IREOSSolution> IS = ireos.evaluateSolutions();
		/* Compute IREOS statistics to this dataset */
		IREOSStatistics stats = ireos.getStatistics();
		/* Print the results */
		for (int i = 0; i < IS.size(); i++) {
			IS.get(i).setStatistics(stats);
			System.out.println("------------------------------------");
			System.out.println("Solution: " + solutions[i]);
			System.out.println("Adjusted IREOS: "
					+ IS.get(i).getAdjustedIREOS());
			System.out.println("IREOS: " + IS.get(i).getIREOS());
			System.out.println("z-test: " + IS.get(i).zTest());
			System.out.println("t-test: " + IS.get(i).tTest());
			System.out.println("------------------------------------");
		}

	}
}
