This folder contains the main code and some results.

Files ending with the same number are relative to the same simulation (movie,trajectory,speed,#of pili,length of pili).

There are 3 bonus trajectories, from simulations with the same parameters.

SIMULATION CHOICES
-Pili keep retracting after unbinding
-No bundling nor re-elongation
-Whenever the force-balance algebra gives a negative force on one or more pili, they are excluded and the operation is repeated. Such pili will still be binded, but with zero force (thus they will surely not detach from the surface, in that timestep)

SIMULATION PARAMETERS
-Timestep dt=1e-4s
-Transient time 5s, then frames every 0.1s, for 40s

MOVIE
-The movie is in real time: framerate=1/resolution=10fps
-Color code for pili:
--red: binded
--green: unbinded and elongating
--blue (dashed): unbinded and retracting
