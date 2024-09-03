# Summer24 (Tiny ML Independent Study
 This is the respository for my independent study at UTSA with Drs. Xie and Pan. The project is based around https://tinymlcontest.github.io/TinyML-Design-Contest/Problems.html and links to their repositories are found on that page (https://github.com/tinymlcontest/tinymlcontest2022_demo_example/).

 ## Overview

 The tinymlcontest2022_demo_example folder is mostly the same as the original. Generated .bat files will need to be run from here.

 In the models folder, there are now two models. The program will not run unless there is a "model_1.py" file available. Since I have two models (one with an avgpool layer to handle shorter input sizes), my automated .bat (sorry it's CMD) renames the desired model before calling the main .py files and then reverts that name when completed. This ensures I'm always running the desired model. Sometimes things get stuck and it's because some part of this renaming failed, just manually set or reset the filename as needed.

 Results are saved in the results folder. The best place to look for readable and verified results will be my Google Sheet here: https://docs.google.com/spreadsheets/d/1QxSq8h2buYRQKXKKzqlOlNe0SRCejuz_MVQv7s9Tgtg/edit?usp=sharing. There are four sheets - one for raw lines of data for the original unmodified model (that only accepts 1250 long inputs), and a sheet for the avgpool model's results (this can accept inputs as small as 98 points). These are each summarized for quicker viewing via Pivot Tables onto "summary" tabs.

## Playground Folder

 The playground folder has a mix of helper files and at this time. Most of my jupyter notebooks that I was playing with are here. The original data came as data.rar that can hopefully be found here: https://drive.google.com/drive/u/0/mobile/folders/1N-zT2p7pQck3uJjO6ORFFyoqeRpSIS3a?sort=14&direction=d . This should be unpacked (but not uploaded here due to GitHub limits) to a folder called 'tinyml_contest_data_training/' in the root folder. 
The file you will want to see most is the testrunner notebook.

## Testrunner Notebook

This is how I'm preparing runs. The first section is imports, declarations, and functions. Note that the getFullTrainSignalMatrix() won't work because GitHub blocks large files (this is our entire 400MB or so dataset it's trying to prepare). You can see what this should be in my "lab_data_driven_sensor_placement" notebook. It's just an np.stack() of all the files pointed to by the train_indices.csv file loaded into numpy with np.loadtxt().

### SVD

You can run these cells to process all of the original data into truncated and reconstructed versions. Choose your number of ranks of the SVD desired 'r' and number of "sensors" desired 'p'. The process function performs the SVD and QR pivoting on the input training data and saves the relevant values / matrices to a .pkl. The prepare then uses this .pkl to actually "sense" and "reconstruct" the data, saving them to respective folders.

NOTE: because files larger than 100MB, a few .pkls aren't there for now.

ref: https://www.researchgate.net/publication/312947855_Data-Driven_Sparse_Sensor_Placement_for_Reconstruction 

### Random Dropout (Static) Mask

This creates one mask 1250-points long of boolean values where each point has a 'keep_percentage' chance of being "True" (i.e., keep this point). We then apply that mask to the original data and save the truncated data to a folder. This means every file has the same relative information missing (in a simple example of data with 5 points and a 60% keep rate, ~two random positions from each file will get dropped. Perhaps: "0, \_, \_, 3, 4".)

### Random Dropout

A mask is made for each file instead of one mask for the whole set, so they're really just each randomly messed up. Different values will be kept from each and each will have a range of lengths centered around 1250 * keep_percentage.

### FFT

A very simple np.ffft.rfft() call on each data point. It looks at the strongest peaks in the results for all files and creates a mask to just keep those points. The data is then processed (with the option to remove/truncate unwanted frequences entirely or leave them as zeros) and saved with a .bat created.

### Wavelet

We transform each file with some wavelet. A lot of decisions made here were informed by exploration in the "explorefftwavelet" notebook (levels of decomposition, determining which level of coefficients to keep, etc). There are two ways to run this, as a simple single-pass discrete wavelet transform, or as something with many levels of decomposition. Choose which cell you want to run.

### Short Time Length

Each sample is 5 seconds long at 250Hz. This section allows us to just take the first half (2.5sec or 625 points), or first second (250 points, etc). This is trivial to do in the avgpool model with the --size parameter. The original model needs padding, hence why this section was made. I could also just change the source code of the train/test files for this, but want to leave those files as close to original as possible just to be safe.