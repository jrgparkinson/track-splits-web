from stravalib.client import Client
import requests
import numpy as np
import numpy.matlib
import operator
import math

class TrackFit:
    # Physical constants
    L = 84.39 # metres, length of athletics track
    R = 36.5 # metres, radius of bend

    R_earth = 6371e3 # metres, earth's mean radius
    lat_to_metres = 110.574e3 # metres/degree, approx conversion of 1 degree to metres
    
    def __init__(self, latlng = None, times = None, race_distance=3000, track_length=400, laps = None, lap_index = -1,  verbosity=0):
        self.latlng = latlng
        self.times = times
        self.race_distance = race_distance
        self.track_length = track_length
        self.verbosity = verbosity
        self.D = None
        self.phi = 0
        self.not_a_track_warning = False
        self.laps = laps
        self.lap_index = lap_index
        
        # If lap_index is defined (>0), get the subset of latlngs and times we want
        if lap_index > 0:
            chosenLap = laps[lap_index-1]
            start_index = chosenLap['start_index']
            
            # Take an extra time point if possible
            if start_index >= 0:
                init_time = self.times[start_index-1]
            else:
                init_time = self.times[start_index]
            
            if lap_index < len(laps):
                end_index = laps[lap_index]['start_index']
                self.latlng = self.latlng[start_index:end_index]
                self.times = self.times[start_index:end_index] 
            else:
                self.latlng = self.latlng[start_index:]
                self.times = self.times[start_index:] 
                
            for t_i in range(0, len(self.times)):
                self.times[t_i] = self.times[t_i] - init_time
        
        if len(self.latlng) > 0:
            self.compute_fit()
            
    def compute_fit(self):
        #Take lat lng and map onto an athletics track

        #Get the track orientation
        self.get_track_angle()
      
        if self.verbosity > 0:
            print('Track angle: ' + str(self.phi) + ' radians')

        # Get the track centre
        [lat_centre, lng_centre] = self.get_centre(self.latlng)

        if self.verbosity > 0:
            print('Track centre: (' + str(lng_centre) + ', ' + str(lat_centre) + ') \n')

        # Define the co-ordinates of a track with centre (lat_centre, lng_centre) and orientation phi

        # First, get a track with centre (0,0) and phi = 0, then rotate and translate appropriately
        [fit_xy, d] = self.defaultTrack() # NOTE: this is in metres

        #fit_coords = fit_coords/lat_to_metres # Convert to lat/lng
        

        # Now rotate by phi using matrix multiplication
        rotation = np.array([[np.cos(self.phi), -np.sin(self.phi)], [np.sin(self.phi), np.cos(self.phi)]])
        
        if self.verbosity > 0:
            print('Rotation matrix: ' + str(rotation))
            print('Multiply rotation matrix (' + str(rotation.shape) + ') by coords matrix (' + str(fit_xy.shape) + ') \n')
            
        # Should rotate first!
        rotated_xy = np.dot(rotation, fit_xy)  
        
        # Now convert to latlng
        fit_latlng =  self.metres_to_latlng(lat_centre, rotated_xy)
        
    
        # Finally, shift centre position
        [fit_lat_c, fit_lng_c] = self.get_centre(fit_latlng)
        
        if self.verbosity > 0:
            print('Current fit centre: (' + str(fit_lng_c) + ', ' + str(fit_lat_c) + ') \n')

        lng_offset = lng_centre - fit_lng_c
        lat_offset = lat_centre - fit_lat_c

        offset_latlng = np.array([[lat_offset], [lng_offset]])

        fit_latlng = fit_latlng + offset_latlng # Apply offset using broadcasting

        [new_fit_lat_c, new_fit_lng_c] = self.get_centre(fit_latlng)
        
        if self.verbosity > 0:
            print('Current fit centre: (' + str(new_fit_lng_c) + ', ' + str(new_fit_lat_c) + ') \n')


        # Now want some visualisation
        # Plot original GPS data and fit coords on same axis
        # Matplotlib is playing up so write out data to a file and load in matlab        
        # F_fit = open('trackFit.csv', 'w')
        # fit_latlng = np.transpose(fit_latlng)
        # for coord in fit_latlng:
            # F_fit.write(str(coord[0]) + ',' + str(coord[1]) + '\n')
        # F_fit.close()

        
        fit_shape = fit_latlng.shape
        GPS_shape = self.latlng.shape
        if self.verbosity > 0:
            print('Fit latlng: ' + str(fit_shape) + ', GPS latlng: ' + str(GPS_shape))
            print('Fit min dim: ' + str(min(fit_latlng.shape)) + ', GPS min dim: ' + str(min(self.latlng.shape)))

        # Fit each lat/lng to the nearest point on the track
        # Check they have the correct shapes first
        
        
        fit_length =  max(fit_latlng.shape)
        latlng_length = max(self.latlng.shape)
        
        if not fit_shape[0] == fit_length:
        #if not ((fit_shape[0] == GPS_shape[0]) or (fit_shape[1] == GPS_shape[1])):
            fit_latlng = np.transpose(fit_latlng)
            
        if self.verbosity > 0:
            print('Fit latlng: ' + str(fit_latlng.shape) + ', GPS latlng: ' + str(self.latlng.shape))
  
            
        
        d_fit = []
        
        j = 0
        distances = np.zeros(fit_length)
        errors = []
        d_fit = np.zeros(latlng_length)
        for ll in self.latlng:
            for i in range(0, fit_length):
                fit_ll = fit_latlng[i]
                distances[i] = np.dot(fit_ll-ll, fit_ll-ll)

            min_index, min_value = min(enumerate(distances), key=operator.itemgetter(1))
            d_fit[j] = d[min_index]
            errors.append(np.sqrt(min_value))

            j = j + 1

        self.fit_error = np.mean(np.array(errors))*self.lat_to_metres
        
        # Save this to the object so we can retrieve it elsewhere
        self.fit_latlng = fit_latlng
        
        self.d_fit = d_fit
        #print('d_fit: ' + str(d_fit))

        self.D = d_fit -  self.get_start_location()
        #self.D = d_fit - d_fit[0] # Make sure our total distance starts at 0!
        for i in range(1, max(self.D.shape)):
            if d_fit[i-1] > d_fit[i]:
                self.D[i:] = self.D[i:] + self.track_length
                
        


    def is_good_fit(self, tolerance=15.0):
        if self.fit_error < tolerance:
            return True
        else:
            return False

    def compute_splits(self, split_distances):
        split_distances = np.array(split_distances)

        total_times = np.empty(split_distances.shape)
        split_times = np.empty(split_distances.shape)
        prev_time = 0
        
        D_arr = np.array(self.D)
        t_arr = np.array(self.times)
        
        speeds = (D_arr[1:] - D_arr[:-1])/(t_arr[1:] - t_arr[:-1])
        mean_speed = np.mean(speeds)

        for i in range(0, len(split_distances)):
            total_times[i] = np.interp(split_distances[i], self.D, self.times)
            
            # A more accurate calculation of the time for the finish would be extrapolation from the previous two values
            if split_distances[i] == self.race_distance:
                closest_D = min(self.D, key=lambda x:abs(x-split_distances[i]))
                where_indices = np.where(self.D == closest_D) # self.D.where(closest_D)
                
                prev_D_i = where_indices[0][0]
                final_speed = speeds[prev_D_i-1]
                
                # If the final speed is sufficiently small do extrapolation
                if final_speed < 0.8 * mean_speed:
                    
                    
                    # If this waypoint is beyond our split, take the previous waypoint
                    if closest_D >= split_distances[i]:
                        prev_D_i = prev_D_i - 1
                        
                             
                    # Go back another waypoint to be a bit more careful
                    #prev_D_i = prev_D_i - 1
                        
                    # Now extrapolate
                    #speed = (self.D[prev_D_i] - self.D[prev_D_i-1])/(self.times[prev_D_i] - self.times[prev_D_i-1])
                    #time_since_prev_split = (split_distances[i] - self.D[prev_D_i])/speed
                    #total_times[i] = self.times[prev_D_i] + time_since_prev_split
                    
                    # Or, try polyfit
                    order = 1 # linear, quadratic, cubic etc.
                    
                    if prev_D_i > order:
                        # if there's enough points for extrapolation...
                        
                        time = np.array(self.times[prev_D_i-(order):prev_D_i])
                        dist = np.array(self.D[prev_D_i-(order):prev_D_i])
                        z = np.polyfit(dist, time, order)
                        
                        p = np.poly1d(z)
                        total_times[i] = p(split_distances[i])
            
            
            split_times[i] = total_times[i]-prev_time
            prev_time = total_times[i]
            
            
        splits = []
        for i in range(0, len(split_distances)):
            thisSplit = {'distance': split_distances[i], 'time': total_times[i], 'split_time': split_times[i]}

            thisSplit['total_mins'] = math.floor(total_times[i] / 60)
            thisSplit['split_mins'] = math.floor(split_times[i] / 60)

            total_secs = total_times[i] - thisSplit['total_mins']*60
            thisSplit['split_secs'] = split_times[i] - thisSplit['split_mins']*60
            
            format_str = '{:04.1f}'
            
            thisSplit['total_secs'] = format_str.format(total_secs)
            thisSplit['split_secs'] = format_str.format(thisSplit['split_secs'])
            
            #thisSplit['total_mins'] = format_str.format(thisSplit['total_mins'])
            #thisSplit['split_mins'] = format_str.format(thisSplit['split_mins'])
            
            
            splits.append(thisSplit)

        if self.verbosity > 0:
            print('distance (m)  | total time (s)  | split time (s)\n')
            for split in splits:
                print(str(split['distance']) + '          |  ' +str(split['total_mins'])+ ':' +str(split['total_secs'])+'       |  '+str(split['split_mins'])+':'+str(split['split_secs']))

                
        # plt.plot(latlng)
        # plt.plot(fit_coords)
        # plt.xlabel('Latitude')
        # plt.ylabel('Longitude')
        # plt.legend('GPS data', 'Fitted data')
        #
        # plt.show()

        return splits

       
     
    def get_start_location(self):
        
        # By knowing the race distance and track length we know the start location could
        # be one of two places, each 200m apart. 
        
        first_fit = self.d_fit[0]
        print('Current start: ' + str(first_fit))
        
        loc = [self.track_length - np.mod(self.race_distance, self.track_length)]
        loc.append(loc[0]+self.track_length/2)
        
        if loc[1] == 0:
            loc.append(self.track_length)
        
        closest_loc = min(loc, key=lambda x:abs(x-first_fit))
        
        print('Using ' + str(closest_loc))
        return closest_loc
       
    

    # Gradient defined as delta(component 1)/delta(component 0)
    def gradient(self, track, i):
        deltaY = track[i+1][1] - track[i-1][1]
        deltaX = track[i+1][0] - track[i-1][0]
        
        grad = deltaY/deltaX

        return grad

    # Get the centre of a track, from noisy data
    def get_centre(self, thisLatLng):

        # First make sure we're indexed correctly
        
        s = thisLatLng.shape
               
        if s[1] > s[0]:
            thisLatLng = np.transpose(thisLatLng)

        latitude  = [l[0] for l in thisLatLng]
        longitude = [l[1] for l in thisLatLng]

        sorted_lat = sorted(latitude)
        sorted_lon = sorted(longitude)

        # We're going to take the N largest/smallest values in each direction, then
        # average these to determine the maximum extents of the track.
        # Specify N here:
        N = 5
        min_lat = np.mean(sorted_lat[:N])
        max_lat = np.mean(sorted_lat[-N:])

        min_lon = np.mean(sorted_lon[:N])
        max_lon = np.mean(sorted_lon[-N:])

        lat_middle = np.mean([min_lat, max_lat])
        lon_middle = np.mean([min_lon, max_lon])

        return [lat_middle, lon_middle]


    def get_track_angle(self):
        i=1
        m=[]
        
        # Convert to metres first, maybe?
        track = self.latlng
        
        #track = self.latlng_to_metres(self.latlng[0][0], self.latlng)
        
        numPoints = track.__len__()
        
        # Need to account for differences in horizontal/vertical distances
        # when computing the gradient
        reference_lat = track[0][0]
        [m_per_deg_lat, m_per_deg_lon] = self.metres_to_latlng_conversion(reference_lat)
        gradient_scale = m_per_deg_lon / m_per_deg_lat
        
        angles = []

        while i<(numPoints-2):
            # Get d(lng)/d(lat) or dy/dx
            gradient = self.gradient(track, i)*gradient_scale
            angles.append(np.arctan(1/gradient))
            if np.isfinite(gradient):
                m.append(1/gradient)
                
            i = i+1


        angles_abs = [abs(x) for x in angles]
        m_abs = [abs(x) for x in m]
        median = np.median(m_abs)
        stdev = np.nanstd(m_abs)
        
        
        
        median_angle = np.median(angles_abs)
        stdev_angle = np.nanstd(angles_abs)
        
        # Compute average angle using histogram. Don't use abs.
        num_bins = round(numPoints/5)
        frequency, bin_edges = np.histogram(angles, num_bins)
        
        bins_mean = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(frequency))]
        
        mode_i = np.argmax(frequency)
        mode_angle = bins_mean[mode_i]
        
        self.bins_freqs = []
        for i in range(0, len(bins_mean)):
            self.bins_freqs.append([bins_mean[i], frequency[i]])
        
        #if isnan(stdev):
        #    stdev = median

        #print(mean)
        # This is the fraction of the standard deviation we will allow from the median value
        tolerance = 0.1
        tolerance_radians = np.pi / 100

        accepted_gradients = []
        accepted_angles = []

        for grad in m:
            if abs((abs(grad) - median)/stdev) < tolerance:
                accepted_gradients.append(grad)
                
        for a in angles:
            if abs(a - mode_angle) < tolerance_radians:
                accepted_angles.append(a)

        self.accepted_angles = accepted_angles
        self.angles = angles
        
        
        self.accepted_angles_limits = [min(accepted_angles), max(accepted_angles)]
        
        
        straight_gradient = np.mean([abs(x) for x in accepted_gradients])
        self.phi = np.arctan(straight_gradient)
        #print(straight_gradient)

        # TODO: need to determine if this angle should be positive or negative
        # Depends on which straight the home straight 
        
        straight_angle = np.mean([abs(x) for x in accepted_angles])
        straight_angle = np.mean(accepted_angles)
        self.phi = straight_angle
        
        
        
        # Get the final gradients
        #if len(accepted_angles) > 1:
        #    final_sign = np.sign(accepted_angles[-2])
        #    self.phi = final_sign*abs(self.phi)
            
        
        #self.phi = -self.phi
        #if self.lap_index == 2:
        #    accepted_angles[10000000000000]
            
        
        if np.isnan(self.phi):
            self.phi = 0
        
        


    def defaultTrack(self):
        numFitPoints = int(self.track_length)
        fit = np.zeros([numFitPoints, 2])
        distances = range(0, numFitPoints)
        for d in distances:
            if d < np.pi*self.R:
                fit[d]  = [self.L + self.R*(1+np.sin(d/self.R)), self.R*(1+np.cos(d/self.R + np.pi))]
            elif d < np.pi*self.R + self.L:
                fit[d] = [self.L+self.R-(d-np.pi*self.R),  2*self.R]
            elif d < 2*np.pi*self.R + self.L:
                fit[d] = [self.R*(1-np.sin((d-self.L-np.pi*self.R)/self.R)), self.R*(1+np.cos((d-self.L-np.pi*self.R)/self.R))]
            else:
                fit[d] = [self.L+self.R - (self.track_length-d), 0]

        fit = np.transpose(fit)
        return [fit, distances]
        
        
    def metres_to_latlng_conversion(self, reference_lat):
        reference_lat = reference_lat*(np.pi/180) # Convert to radians
        m_per_deg_lat = 111132.92 - 559.822 * np.cos( 2 * reference_lat ) + 1.175 * np.cos( 4 * reference_lat) - 0.0023*np.cos(6*reference_lat)
        m_per_deg_lon = 111412.84 * np.cos ( reference_lat ) - 93.5*np.cos(3*reference_lat) + 0.118*np.cos(5*reference_lat)
        
        return [m_per_deg_lat, m_per_deg_lon]

    # Scales X, which should be a 2xN vector of co-ordinates,
    # Appropriately for the given reference latitude
    def metres_to_latlng(self, reference_lat, X):
        # Formula from https://en.wikipedia.org/wiki/Geographic_coordinate_system
        [m_per_deg_lat, m_per_deg_lon] = self.metres_to_latlng_conversion(reference_lat)

        s = X.shape

        # Ensure the 0 index is N (so the 1 index is 2)
        if s[1] > s[0]:
            X = np.transpose(X)

        arr = [[xy[1]/m_per_deg_lat, xy[0]/m_per_deg_lon] for xy in X]
        

        latlng = np.array(arr)

        # Tranpose again so we can do matrix multiplication for rotation
        latlng = np.transpose(latlng)

        return latlng
        
    def latlng_to_metres(self, reference_lat, latlng):
        # Formula from https://en.wikipedia.org/wiki/Geographic_coordinate_system
        [m_per_deg_lat, m_per_deg_lon] = self.metres_to_latlng_conversion(reference_lat)

        print('latlng to metres, reference lat = ' + str(reference_lat))
        
        s = latlng.shape

        # Ensure the 0 index is N (so the 1 index is 2)
        if s[1] > s[0]:
            latlng = np.transpose(latlng)

        # array of y, x values
        arr = [[ll[0]*m_per_deg_lat, ll[1]*m_per_deg_lon] for ll in latlng]

        yx = np.array(arr)

        # Tranpose again so we can do matrix multiplication for rotation
        xy = np.transpose(yx)

        return xy

   