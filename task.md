The goal is to create a pipeline for synthetic data generation.
Pipeline has 3 steps:
1. generate synthetic data
2. perform QC
3. push to HF

## 1. generate synthetic data
This should be used - https://github.com/lukasugar/NV-Generate-CTMR/blob/test_generation
Check the recent commits to figure out how it works.

## 2. QC
QC should be done with SynthSeg, like in `src/preprocessing/README.md`.
SynthSeg generates some QC scores. Those scores should be used to filter out bad quality outputs.

## 3. Push to HF
This step should do nothing initially. Just print "Pushing to HF not implmeneted yet". But it should be obvious where in the pipeline this step belongs.


# Notes
The pipeline should be parametrized: what number of images to generate, which modalities, etc.
The treshold for QC should be specified. If not, then no filtering is applies.

the code should be put into `src/synthetic_pipeline`. 
That folder should have a readme file that explains how to setup and run synthetic pipeline.

There should also be an sbatch script in `scripts` which can be used to submit the pipeline job on slurm.

Don't submit to slurm, trust that it will work.

Write tests only if you want to test the code when you implement it. The final code shouldn't contain tests.

# Important
Don't make any assumptions about implmentation. Ask the user brainstorming questions, to understand the requirements and the plan.


