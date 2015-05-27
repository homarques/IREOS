IREOS (Internal, Relative Evaluation of Outlier Solutions)

Implementation by Henrique O. Marques < hom@icmc.usp.br >

Original paper:
H.O. Marques, R.J.G.B. Campello, A. Zimek and J. Sander. On the Internal Evaluation of Unsupervised Outlier Detection. In Proceedings of the 27th International Conference on Scientific and Statistical Database Management (SSDBM), San Diego, CA, 2015.

Included in this distribution is an example data set (WBC_withoutdupl_norm) which consists of 223 objects, each with 9 attributes, this data set is a publicly available real world datasets from the UCI repository modified following the procedure described in the paper.
Also included are 11 outlier solutions of the data set (folder solutions) which the name of the file is the number of outliers correctly labeled according to the ground truth.

Usage:
======

-Create dataset model using dataset file (plain text format, like dataset example provided), the dataset size and cost weight used by KLR:
	-SVMExamples dataset = new SVMExamples(BufferedReader reader, int size, double c);

-Create the list of solutions to be evaluated, each list element must be a vector that represent if the correspondent dataset element is an outlier (1) or an inlier (-1):
	-List<int[]> ireosSolutions

-Initialize IREOS using the dataset and the list of solutions to be evaluated:
	-IREOS ireos = new IREOS(dataset, ireosSolutions);

-Set Gamma Maximum:
	-ireos.setGammaMax(double gammaMax);

-Or find a Gamma Maximum:
	-ireos.findGammaMax();

-Set the number of values that Gamma will be discretized:
	-ireos.setnGamma(int nGamma);

-Discretize Gamma from 0 to Gamma Maximum into the number of values established, the discretization can be 0 (linear), 1 (quadratic), 2 (logarithmic) or 3 (logarithmic (legacy):
	-ireos.setGammas(int scale);

-Set the maximum clump size:
	-ireos.setmCl(int mCl)

-Evaluate the solutions:
	-List<IREOSSolution> evaluatedSolutions = ireos.evaluateSolutions();

-Return the IREOS index:
	-evaluatedSolutions.get(i).getIREOS();


-Optionally:
============

-Compute IREOS statistics to the dataset in order to perform statistical tests and to adjust the index for chance:
	-IREOSStatistics stats = ireos.getStatistics();

-Set the IREOS statistics to the solution
	-evaluatedSolutions.get(i).setStatistics(stats);

-Return the Adjusted IREOS index:
	-evaluatedSolutions.get(i).getAdjustedIREOS()

-Return the z-test:
	-evaluatedSolutions.get(i).zTest()

-Return the t-test:
	-evaluatedSolutions.get(i).tTest()