# Platinum: Nearest RGBDL Sphere Request

Python executable providing the nearest RGB-DL sphere according to a given query.

## Requesting the nearest sphere

Launch the python script like this:
```
./request.py path_to_the_query
```
Optionally defined the output folder:
```
./request.py path_to_the_query --out_path output_dir --out_file name_of_the_output
```
The simimarity ranking result is written in a *csv* file where each line refere to a pair {sphere_idx, similarity_score}, ranked in descending order (according to the similariy score).
It may have mutiple occurance of the same sphere idx, if the original panorama have been spilted in the offline stage.

## Building database

This offline step have to been performed when the spheres database is updated, in order to mainten proper signature for the similarity comparison.
Launch the python script like this:
```
./update_db.py graph_csv_file
```
Input csv file must have the following architecture:
```
# Comment ...
# Comment ...
0;x_i;y_i;path_to_rgb_i;path_to_depth_i;path_to_sem_i
1;x_j;y_j;path_to_rgb_j;path_to_depth_j;path_to_sem_j
...
```
You can precise the absolute path of the graph file location like this:
```
./update_db.py graph_csv_file --root path_to_data_folder
```

## Miscellaneous
To display help and all parameters, use *-h*
