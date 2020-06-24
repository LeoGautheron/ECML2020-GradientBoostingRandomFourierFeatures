## Dependencies
Python3.7.3 and some python3 libraries:
 - numpy (v1.16.2 used)
 - sklearn (v0.22.2.post1 used)
 - scipy (v1.2.1 used)
 - lightgbm (v2.3.1 used)
 - matplotlib (v3.0.3 used)

## Content
folder 'datasets'
 - contain the 16 datasets used in the experiments

folder 'experiment-toy-LGBM-GBRFF2':
  - contain the source code to reproduce the experiment on toy datasets
  where LGBM and our proposed method GBRFF2 are compared
 - the experiment is launched with 'python main.py seed' which apply the
   algorithms on the generated datasets with a fixed seed, and store the
   results in a file. We executed this script with seeds from 1 to 20.
 - the file 'plotBoundaryAccuracy.py' load the result files, and produce the
   figure used in the paper (some modifications were made to produce the exact
   same figure in the paper)

folder 'experiment-progression-PBRFF-GBRFF1-GBRFF2':
  - contain the source code to reproduce the experiments showing
    the progression between PBRFF, GBRFF0.5, GBRFF1, GBRFF1.5 and GBRFF2
 - the experiment is launched with 'python main.py seed' which apply the
   algorithms on the 15 first datasets from the dataset folder with a
   fixed seed, and store the results in a file. We executed this script with
   seeds from 1 to 20.
 - the file 'plot.py' load the result files, and produce the figures used in
   the paper

folder 'experiment-latex-array-results':
  - contain the source code to reproduce the experiments producing a latex
    array comparing BMKR, GFC, PBRFF, GBRFF1, LGBM and GBRFF2
 - the experiment is launched with 'python main.py seed' which apply the
   algorithms on the 15 first datasets from the dataset folder with a
   fixed seed, and store the results in a file. We executed this script with
   seeds from 1 to 20.
 - To launch the experiment on the last dataset "bankmarketing" the file
   "datasets.py" needs to be modified (at the very bottom of the file)
 - the file 'latex.py' load the result files, and produce a .tex file
   containg a tabular used in the paper

folder 'experiment-computation-time':
  - contain the source code to reproduce the experiments comparing the
    computation time of the methods on an artificially generated dataset having
    an increasing number of examples
 - the experiment is launched with 'python main.py seed' which apply the
   algorithms on the generated datasets until the time limit is reached with a
   fixed seed, and store the results in a file. We executed this script with
   seeds from 1 to 20.
 - the file 'plot.py' load the result files, and produce the figure used in
   the paper

folder 'experiment-toy-loss-placing-landmark-GBRFF1':
  - contain the source code to reproduce the toy experiment where we show
    on the two moons datasets that learning a single scalar is equivalent to
    learn a full landmark vector to minimize our loss function
 - the file 'main.py' launch the experiment and produce the figure used in
   the paper

folder 'experiment-bars-GBRFF1':
  - contain the source code to reproduce the experiment comparing GBRFF1
    with different total number of random features T*K in the form of a
    barchart
 - the experiment is launched with 'python main.py seed' which apply the
   algorithms on the 15 first datasets from the dataset folder with a
   fixed seed, and store the results in a file. We executed this script with
   seeds from 1 to 20.
 - the file 'bars.py' load the result files, and produce the figure used in the
   paper
