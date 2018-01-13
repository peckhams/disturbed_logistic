
# S.D. Peckham
# May 10-11, 2011
# May 12,13, 2011: Minor tweaks
# May 16, 2011:  PRINT_TO_FILE option
# May 20, 2011: run_type in __init__()
# Jan 15, 2013: New threshold condition (digamma function)
# Aug 2017:  Ability to compute empirical distributions.
# Jan 2018:  save_steady_pdf() function and cleanup.

#-----------------------------------------------------------------
# Notes: To run this model for 20 disturbances and then plot
#        N(t) vs. t do the following.
#        Open a terminal window.
#        Change to the directory with this file.
#        Type:  python  (to open a python shell)
#        Type:  import logistic_growth as lg
#        Type:  lg.unit_test()
#
#        To change the model parameters, edit the function
#        "read_cfg_file().  Model parameters are currently
#        hard-wired in the "__init__()" and "read_cfg_file()"
#        method functions of the logistic_growth class.  Later,
#        "read_cfg_file()" will read them from a configuration
#        file (as text).
#
#        Note that the code is written to be mostly compliant
#        with the Basic Model Interface (BMI), which was
#        developed by the NSF-funded CSDMS project.
#        See Peckham et al. (2013)

#        To run the model for 10 million timesteps to compute
#        an empirical invariant distribution, do this:
#        Type:  lg.get_steady_pdf()
#        The invariant distn. is a delta function for r < r2.
#        This will take a few minutes to run.
#
#        To plot a previously computed & saved empirical pdf,
#        do the following:
#        Type:  epdf_file = 'your_epdf_file.txt'
#        Type:  lg.read_steady_pdf()
#        You will want to edit paths set by os.chdir().
#        Note that get_steady_pdf() saves epdf by default.
#
#        For more information on this model, see:
#        Peckham, Waymire and Leenheer (2017) Critical thresholds
#        for eventual extinction in randomly disturbed population
#        growth models (in review), J. of Math. Bio.
#
#        The three critical threshold conditions are:
#        (1) r < r1 = lambda * [1 - E(X)]
#            I1 = (r - r1)
#
#        (2) r < r2 = lambda * E[-ln(X)]
#            I2 = (r - r2),
#
#        (3) r < r3 = lambda * [E(1/X) - 1]
#            I3 = (r - r3)
#
#-----------------------------------------------------------------

# unit_test()
# get_steady_pdf()               (8/19/17)
# plot_steady_pdf()              (8/19/17)
# plot_steady_pdf_set()          (1/3/18)
# save_steady_pdf()              (1/3/18)
# read_steady_pdf()              (1/3/18)
#
# get_path_avg()      (experimental, not ready)
# plot_path_avg()     (experimental, not ready)
#
# class logistic_model()
#
#    __init__()
#    read_cfg_file()
#    get_X_values()
#    get_T_values()
#    get_T_values2()    (7/14/13)
#
#    run_model()
#    initialize()
#    update()
#    update_T()
#    update_X()
#    update_N()
#    update_time()
#    get_plot_info()
#    plot_N_vs_t()
#    finalize()
#
# monte_carlo()                     ### very old, may not work.
# monte_carlo_print_to_file()       ### (not ready yet)
# monte_carlo_print_lambda_vs_mu()  ### (not ready yet)
# plot_data_from_file()
#
#-----------------------------------------------------------------

import numpy as np
import numpy.random
from   scipy.special import psi as digamma    ### (1/15/13)
import matplotlib.pyplot as pyplot
import matplotlib.cm as cmaps
import time
import os    # (for plot_data_from_file())

#-----------------------------------------------------------------
def unit_test():
    
    model = logistic_model( n_target=20 )
    model.run_model()

#   unit_test()
#-----------------------------------------------------------------
def get_steady_pdf( SAVE=True, PLOT=True ):
    
    start_time = time.time()
    model = logistic_model()
    
    model.SILENT = False
    model.PLOT   = False
     
    model.run_model()

#     print 'First 20 N values ='
#     print model.N[0:20]
#     print ' '
    
    #-----------------------------------------------------------    
    # Skip first 100 values, build empirical pdf from the rest
    #-----------------------------------------------------------
    # model.N is the vector of discrete-time N values to bin
    # The N array is declared larger than actually used.
    #-----------------------------------------------------------
    ## n_bins = 50
    n_bins = 100
    K = model.K
    vals = model.N[100: model.jump_index + 1]
    h = np.histogram( vals, bins=n_bins, range=(0,K),
                      density=True )

    ## print '###### model.jump_index =', model.jump_index
    ## print '###### vals.size =', vals.size
    
    #-------------------------    
    # Plot the empirical pdf
    #----------------------------------------------------------
    # Agrees with closed-form result for Uniform disturbances
    #----------------------------------------------------------
    hist = h[0]
    bin_edges = h[1][1:]  # (otherwise bigger by 1)
    ## print 'shape(h[0]) = ', h[0].shape
    ## print 'shape(h[1]) = ', h[1].shape

    #-----------------------------------------
    # Use bin centers instead of right edges
    #-----------------------------------------
    dx = (bin_edges[1] - bin_edges[0])
    bin_edges -= (dx / 2.0)
    
    #---------------------------------------    
    # Plot the steady-state, empirical pdf
    #---------------------------------------
    if (PLOT):
        plot_steady_pdf( bin_edges, hist, model.r, K )

    #------------------------------------------------    
    # Save the steady-state, empirical pdf (12/3/18)
    #------------------------------------------------
    n_events = model.jump_index
    if (SAVE):
        save_steady_pdf( bin_edges, hist, model.beta_a, model.beta_b,
             model.mean_total_n_events, n_events,
             model.r, K, model.N0, model.exp_lambda,
             model.beta_eta, model.beta_m, model.beta_mu, 
             model.r1, model.r2, model.r3, model.I1, model.I2, model.I3)       

#   get_steady_pdf()
#-------------------------------------------------------------
def plot_steady_pdf( x_bin, N_epdf, r, K ):
	
	dx   = (x_bin[1] - x_bin[0])
	mass = (N_epdf.sum() * dx)
	
	print 'Max(pdf)  =', N_epdf.max()
	print 'mass(pdf) =', mass
	print ' '
	
	## label_size = 15
	label_size = 18
	# marker_size = 3
	
	#----------------------------------
	# Method 1. Use matplotlib.pyplot
	#----------------------------------
	## fig = pyplot.figure( figsize=(7.5,5.0), dpi=100, facecolor='w', edgecolor='k')
	fig = pyplot.figure( facecolor='w', edgecolor='k' )
	ax  = fig.add_subplot(111)
	c1  = ax.plot( x_bin, N_epdf )
	## c2  = ax.plot( x_bin, 1.2 * N_epdf)   # this works
	## c3  = ax.plot( x_bin, 1.4 * N_epdf)
	#----------------------------------------
	## c1 = ax.plot( x_bin[1:], N_epdf[1:] )	# exclude 1st bin
	
	# c2 = ax.plot( self.plot_t_vals, self.plot_N_vals, 'r+' )
	# c3 = ax.plot( self.plot_t_jumps, self.plot_N_jumps, 'ro',
	#              markersize=marker_size)
	
	pyplot.setp(ax.get_xticklabels(), fontsize=label_size)
	pyplot.setp(ax.get_yticklabels(), fontsize=label_size)
	title = 'Empirical Invariant Distribution PDF'
	## title += ' (r =' + str(r) + ', K =' + str(K) + ')'
	ax.set_title( title, fontsize=label_size)
	ax.set_xlabel( '$x$', fontsize=label_size )
	ax.set_ylabel( '$f_N(x)$', fontsize=label_size )

	#----------------------------------------------	
	# Need this to set y-axis limits, when needed
	#----------------------------------------------
	pyplot.axis([-1.0, K + 1.0, 0, 3.0/K])   #############
	## pyplot.axis([0, 10.5, 0, 1.0])   #############
	## pyplot.axis([0, 10.0, 0, 0.05])  #############
	ax.set_autoscale_on( False )
	
	pyplot.show()
	
#   plot_steady_pdf()
#-------------------------------------------------------------
def plot_steady_pdf_set():
	
	## label_size = 15
	label_size = 18
	
	#----------------------------------
	# Method 1. Use matplotlib.pyplot
	#----------------------------------
	## fig = pyplot.figure( figsize=(7.5,5.0), dpi=100, facecolor='w', edgecolor='k')
	fig = pyplot.figure( facecolor='w', edgecolor='k' )
	ax  = fig.add_subplot(111)

	#--------------------------------------	
	# Add several epdf curves in one plot
	#--------------------------------------		
	files = ['r=0.44', 'r=0.445', 'r=0.45', 'r=0.46', 'r=0.47',
	         'r=0.48', 'r=0.5', 'r=0.52',
	         # 'r=0.53',
	         'r=0.55', 'r=0.6', 'r=0.65', 'r=0.7',
	         'r=0.75', 'r=0.77', 'r=0.8', 'r=0.85', 'r=0.9',
	         'r=1.0', 'r=1.1', 'r=1.2', 'r=1.3', 'r=1.5',
	         'r=1.7', 'r=2.0',
	          # 'r=2.5', 
	         'r=3.0', 'r=5.0', 'r=10.0', 'r=1000.0'] 
	for k in xrange( len(files) ):
	    fname  = 'EPDF_a=3.0_b=2.0_' + files[k] + '.txt'
	    data   = read_steady_pdf( fname, PLOT=False, SILENT=True )
	    x_bin  = data[0]
	    N_epdf = data[1]
	    r      = data[2]
	    K      = data[3]
	    c1     = ax.plot( x_bin, N_epdf )

	#------------------------------------	
	# Make some adjustments to the plot
	#------------------------------------
	pyplot.setp(ax.get_xticklabels(), fontsize=label_size)
	pyplot.setp(ax.get_yticklabels(), fontsize=label_size)
	title = 'Empirical Invariant Distribution PDFs vs. r'
	ax.set_title( title, fontsize=label_size)
	ax.set_xlabel( '$x$', fontsize=label_size )
	ax.set_ylabel( '$f_N(x)$', fontsize=label_size )

	#----------------------------------------------	
	# Need this to set y-axis limits, when needed
	#----------------------------------------------
	pyplot.axis([-1.0, K + 1.0, 0, 0.05])   ########
    ## pyplot.axis([-1.0, K + 1.0, 0, 10.0/K])   ###########
	## pyplot.axis([-1.0, K + 1.0, 0, 3.0/K])   ###########
	## pyplot.axis([0, 10.5, 0, 1.0])   #############
	## pyplot.axis([0, 10.0, 0, 0.05])  #############
	ax.set_autoscale_on( False )
	
	pyplot.show()
	
#   plot_steady_pdf_set()
#-------------------------------------------------------------
def save_steady_pdf(x_bin, epdf, beta_a, beta_b,
                    mean_total_n_events, n_events,
                    r, K, N0, exp_lambda,
                    beta_eta, beta_m, beta_mu, 
                    r1, r2, r3, I1, I2, I3):
                    
	#-------------------------------------------------
	#  Notes:
	#  Open a terminal window.
	#  Change to the directory with this file.
	#  Type:  python  (to open a python shell)
	#  Type:  import logistic_growth as lg
	#  Type:  lg.get_steady_pdf()
	#-------------------------------------------------
	dx = (x_bin[1] - x_bin[0])
		
	prefix   = 'EPDF_'
	astr     = 'a=' + str(beta_a)
	bstr     = 'b=' + str(beta_b)
	suffix   =  astr + '_' + bstr + '.txt'
	out_file = prefix + suffix
	## out_file = 'Logistic_Output1.txt'
	out_unit = open( out_file, 'w' )
	
	#--------------------
	# Write header info
	#--------------------
	out_unit.write('Empirical Invariant Distribution PDF\n\n')
	
	out_unit.write('Parameters for logistic model:\n\n')
	
	out_unit.write('mean_total_n_events = ' + str(mean_total_n_events) + '\n')
	out_unit.write('n_events            = ' + str(n_events) + '\n')	
	out_unit.write('histogram dx        = ' + str(dx) + '\n\n')	
	# out_unit.write('n_reps  = ' + str(n_reps) + '\n\n')
	
	out_unit.write('beta_a   = ' + str(beta_a) + '\n')
	out_unit.write('beta_b   = ' + str(beta_b) + '\n')
	out_unit.write('r        = ' + str(r) + '\n')
	out_unit.write('K        = ' + str(K) + '\n')
	## out_unit.write('b       = ' + str(b) + '\n')
	out_unit.write('lambda   = ' + str(exp_lambda) + '\n')
	out_unit.write('N0       = ' + str(N0) + '\n\n')

	out_unit.write('beta_mu  = E[X]      = ' + str(beta_m)   + '\n')	
	out_unit.write('beta_eta = E[-ln(X)] = ' + str(beta_eta) + '\n')
	out_unit.write('beta_m   = E(1/X)    = ' + str(beta_mu)  + '\n\n')

	out_unit.write('r1  = lam * [1 - E(X)]   = ' + str(r1) + '\n')	
	out_unit.write('r2  = lam * E[-ln(X)]    = ' + str(r2) + '\n')
	out_unit.write('r3  = lam * [E(1/X) - 1] = ' + str(r3) + '\n\n')

	out_unit.write('I1  = r - r1 = ' + str(I1) + '\n')
	out_unit.write('I2  = r - r2 = ' + str(I2) + '\n')
	out_unit.write('I3  = r - r3 = ' + str(I3) + '\n\n')
				   
	#-------------------------------------------
	# Write x_bin and epdf arrays to text file
	#-------------------------------------------
	# print 'x_bin.size =', x_bin.size
	# print 'epdf.size  =', epdf.size
		
	for j in xrange( epdf.size ):
		out_unit.write( str(x_bin[j]) + ' ' + str(epdf[j]) + '\n' )
		
	#-----------------
	# Close the file
	#-----------------
	out_unit.close()
	print 'Finished writing EPDF to ' + out_file + '.'
	print ' '
	
#   save_steady_pdf()
#-------------------------------------------------------------
def read_steady_pdf( epdf_file, PLOT=True, SILENT=False ):

    n_header1 = 8
    n_header2 = 13
    nx = 2
    ny = 100
   
    #------------------------- 
    # Open epdf_file to read
    #-------------------------
    os.chdir( '/Users/peckhams/Desktop/logistic/data' )
    epdf_unit = open( epdf_file, 'r' )
	
    #-----------------------------------------
    # Read the header lines up to main params
    #-----------------------------------------
    for k in xrange( n_header1 ):
        line = epdf_unit.readline()

    #--------------------------
    # Read the main parameters
    #--------------------------
    line   = epdf_unit.readline()
    parts  = line.split("=")
    beta_a = np.float64( parts[1] )
    #--------------------------------
    line   = epdf_unit.readline()
    parts  = line.split("=")
    beta_b = np.float64( parts[1] )
    #--------------------------------
    line   = epdf_unit.readline()
    parts  = line.split("=")
    r      = np.float64( parts[1] )
    #--------------------------------
    line   = epdf_unit.readline()
    parts  =line.split("=")
    K      = np.float64( parts[1] )
    #--------------------------------
    line   = epdf_unit.readline()
    parts  = line.split("=")
    lam    = np.float64( parts[1] )    # lambda is reserved word
    #--------------------------------
    line   = epdf_unit.readline()
    parts  = line.split("=")
    N0     = np.float64( parts[1] )
            
    #-----------------------------------------
    # Read the header lines after main params
    #-----------------------------------------
    for k in xrange( n_header2 ):
        line = epdf_unit.readline()
  
    #------------------------------
    # Read the epdf as a 2D array
    #------------------------------
    data = np.fromfile( epdf_unit, dtype='float64',
                        sep=" ", count=(nx*ny) )
    data = np.reshape(data, (ny, nx))
    ## print 'shape =', np.shape(data)
    x_bin = data[:,0]
    epdf  = data[:,1]
    
    #-----------------
    # Close the file
    #-----------------
    epdf_unit.close()
    if not(SILENT):
        print 'Finished reading EPDF from ' + epdf_file + '.'
        print ' '
    
    #------------------------
    # Option to plot the PDF
    #------------------------
    if (PLOT):
        plot_steady_pdf( x_bin, epdf, r, K )
        
    return (x_bin, epdf, r, K)
	  	               
#   read_steady_pdf()
#-----------------------------------------------------------------
def get_path_avg( n_paths=100 ):
    
    start_time = time.time()
    model = logistic_model( indicator=indicator )
    
    model.SILENT = False
    model.PLOT   = False
    
    path_sum = np.zeros( model.N.size )
    
    for count in xrange( n_paths ):
        model.run_model()
        path_sum += model.N  # (in-place)
        
    path_sum /= n_paths  # (in-place)

    #-----------------------------------    
    # Plot the avg of all sample paths
    #-----------------------------------
    plot_path_avg( t, path_sum )
        
#   get_path_avg()
#-----------------------------------------------------------------
def plot_path_avg( t_vals, path_avg, logistic_r ):
	
	label_size = 15
	# marker_size = 3
	
    ## self.get_plot_info()
        
	#----------------------------------
	# Method 1. Use matplotlib.pyplot
	#----------------------------------
	fig   = pyplot.figure()
	ax    = fig.add_subplot(111)
	c1 = ax.plot( t_vals, path_avg )  ############
	
	# c2 = ax.plot( self.plot_t_vals, self.plot_N_vals, 'r+' )
	# c3 = ax.plot( self.plot_t_jumps, self.plot_N_jumps, 'ro',
	#   			markersize=marker_size)
				  
	pyplot.setp(ax.get_xticklabels(), fontsize=label_size)
	pyplot.setp(ax.get_yticklabels(), fontsize=label_size)
	title = 'Randomly Disturbed Logistic Growth'
	title += ' (r =' + str(self.logistic_r) + ')'
	ax.set_title( title, fontsize=label_size)
	ax.set_xlabel( 'time [years]', fontsize=label_size )
	ax.set_ylabel( 'Population size, N(t)', fontsize=label_size )
	pyplot.show()
        
#   get_path_avg()
#-----------------------------------------------------------------   
class logistic_model():

    #-------------------------------------------------------------
    # def __init__(self, indicator=0.3, n_target=None ):
    def __init__(self, n_target=None ):

        self.seed   = None  # (use default/random seed)
        self.SILENT = False
        self.PLOT   = True
        
        if (n_target is not None):
            self.n_target = n_target
        
        #----------------------------------------
        # With r=1.5, lambda=1.0 and m=2.0, the
        # population is expected to survive.
        #
        # m = (a + b - 1) / (a - 1)
        # a = 1 + (b/(m-1))
        #----------------------------------------
#         self.logistic_r = np.float64( 1.5 )
#         self.logistic_b = np.float64( 0.001 )
#         self.logistic_K = self.logistic_r / self.logistic_b
        
        #----------------------------------------------------
        # Parameters of a Beta distribution
        #     (a > b) => skewed toward x=1
        #     (a < b) => skewed toward x=0
        #----------------------------------------------------
##        self.beta_a = np.float64( 3.0 )
##        self.beta_b = np.float64( 2.0 )
        # This (a,b) gives mu = 3/5, m = 2, eta = 7/12
        # I = (r - lam*eta) = (1.5 - (1*0.583)) > 0
        #----------------------------------------------------
        # This also has mu=3/5, but a much smaller spread
        # around the mean.
        #----------------------------------------------------
##        self.beta_a = np.float64( 21.0 )
##        self.beta_b = np.float64( 14.0 )
        # This (a,b) gives mu = 3/5, m = 17/10 = 1.7.
        #----------------------------------------------------
        # This also has m=2, but a much smaller spread
        # around the mean.
        #----------------------------------------------------
        # self.beta_a = np.float64( 21.0 )
        # self.beta_b = np.float64( 20.0 )
        # This (a,b) gives mu = 2, m = 21/41 = 0.51
        # and eta = 0.52047.

#####################################################################
        #----------------------------------------------------
        # (1/15/13) New parameters to get eta = 1 and then
        # we vary lambda as before to explore threshold.
        # (a < b) => skewed slightly to the left (x=0).
        #----------------------------------------------------
#         self.beta_a = np.float64( 2.0 )
#         self.beta_b = np.float64( 2.63869 )
        # This (a,b) gives eta = 1.0, mu = ???, m = ???
        
        #----------------------------------------------------
        # This has a much smaller spread around the mean.
        #----------------------------------------------------
##        self.beta_a = np.float64( 12.0 )
##        self.beta_b = np.float64( 3.0 )
        # This (a,b) gives mu = 4/5, m = 14/11 = 1.27.

        #-----------------------------------------------------
        # (1/15/13) New threshold condition with eta (a.s.)
        #-----------------------------------------------------
        # Time between disturbances has an Exponential distn.
        #   which corresponds to a Poisson event process.
        #-----------------------------------------------------
        # Changing lambda is the easiest way to change
        # the threshold parameter:  I = r - (lambda * eta).
        # For I=0, lambda = r/eta.
        # If we take (eta = 1) as above, then I=(r - lambda):
        #    (I = 0) => lambda = r
        #    (I < 0) => lambda > r  (certain extinction)
        #    (I > 0) => lamdda < r  (certain invariant distn)
        #-----------------------------------------------------
        ## self.mean_total_n_events = np.float64(30)
        ## self.mean_total_n_events = np.float64(50)
        self.mean_total_n_events = np.float64(100)
        
        #-------------------------------------------
        # Now "indicator" is an argument (1/15/13)
        #-------------------------------------------
        # Hard-wired, for now
        #----------------------
        # run_type = 'above_critical'
        # run_type = 'at_critical'
##        run_type = 'below_critical'
##
##        type_map = {'below_critical':  1.0,
##                    'at_critical':     0.0,
##                    'above_critical': -1.0}
##        indicator = type_map[ run_type ]
##        # factor  = 0.1
##        factor    = 0.3
##        indicator = (indicator * factor)

#         self.indicator = indicator

        #  self.exp_lambda = (self.logistic_r - indicator)
        # print '#### indicator, lambda = ', indicator, self.exp_lambda
        # print ' '
        
    #   __init__()     
    #-------------------------------------------------------------
    def read_cfg_file(self):

        #----------------------------------------------------------
        # Note:  Later, this function will read model para-
        #        meters from a configuration file or CFG file
        #        as key-value pairs.  For now, they are hardwired
        #        here, and the code is edited as necessary.
        #----------------------------------------------------------
        # Note:  set_computed_parameters() is called after this.
        #----------------------------------------------------------        
        # n_target is only set in unit_test().
        #-------------------------------------        
        ## if (self.n_target is not None):
        if (hasattr(self, 'n_target')):
            self.mean_total_n_events = np.float64( self.n_target )     
        else:
            self.mean_total_n_events = np.float64( 10000000.0 )
            ## self.mean_total_n_events = 100000.0
            ## self.mean_total_n_events = 30.0
        
        #------------------------------------------
        # Set the deterministic growth parameters
        #------------------------------------------
#         self.N0    = np.int64( 20 )
#         self.K     = np.float64( 1500 )
#         self.r     = np.float64( 1.5 )  
        #-----------------------------------------------------
        # For comparison to Mathematica plot of computed pdf
        # for steady-state distn with Uniform disturbances.
        # Also set exp_lambda = 0.75.
        #-----------------------------------------------------
        self.N0    = np.int64( 50 )
        self.K     = np.float64( 100 )
        # self.r     = np.float64( 1.5 )
        self.r     = np.float64( 0.85 )
        #------------------------------
        # Reviewer parameters
        #------------------------------
        # self.N0    = np.int64( 10 )
        # self.K     = np.float64( 2000 )
        # self.r     = np.float64( 1.5 )
        #------------------------------
        self.logistic_r = self.r
        self.logistic_K = self.K
        self.logistic_b = (self.logistic_r / self.logistic_K)
        
        #-----------------------------------------------
        # The current model results are for N_min = 0.
        # N_min is now used in "update_N()". (1/15/13)
        # The "Allee effect" is when "N_min > 0".
        # We had N_min = 1 before (1/15/13).
        #-----------------------------------------------
        self.N_min = np.int64( 0 )
        self.N_max = np.int64( self.K )
        
        #------------------------------------------        
        # Option 1: Set lambda, compute indicator.
        # Note:     In paper, p = lambda / r.
        #-------------------------------------------
        self.exp_lambda = np.float64( 0.75 )
        # self.exp_lambda = np.float64( 1.5 )        # Reviewer parameter
        
        #--------------------------------------------        
        # Option 2: Set indicator, compute lambda.
        #--------------------------------------------
        # Note:  Set lambda = -1 to use this option.
        #--------------------------------------------
#         self.exp_lambda = np.int64( -1.0 )    ### FLAG VALUE
#         self.indicator  = np.int64( 0.3 )

        #-------------------------------------------------
        # Set parameters for the the random fraction, X,
        # that survive a disturbance (Beta distribution)
        #-------------------------------------------------
        self.beta_a = np.float64( 1.0 )  # uniform
        self.beta_b = np.float64( 1.0 )       
        #----------------------------------------------------
        # self.beta_a = np.float64( 2.0 )  # right triangle (right)
        # self.beta_b = np.float64( 1.0 )        
        #----------------------------------------------------        
        # self.beta_a = np.float64( 0.5 )  # arcsine distn.
        # self.beta_b = np.float64( 0.5 )        
        #----------------------------------------------------
        # self.beta_a = np.float64( 1.5 )  # semi-elliptic distn.
        # self.beta_b = np.float64( 1.5 )        
        #----------------------------------------------------        
        # self.beta_a = np.float64( 2.0 )  # parabolic distn.
        # self.beta_b = np.float64( 2.0 )        
        #---------------------------------------------------- 
        # self.beta_a = np.float64( 4.0 )  # bell-shape distn.
        # self.beta_b = np.float64( 4.0 )        
        #---------------------------------------------------- 
        # self.beta_a = np.float64( 3.0 )
        # self.beta_b = np.float64( 2.0 ) 
        #----------------------------------------------------
        # self.beta_a = np.float64( 2.0 )
        # self.beta_b = np.float64( 3.0 )
        #----------------------------------------------------
        # self.beta_a = np.float64( 1.0 )  # right triangle (left)
        # self.beta_b = np.float64( 2.0 )
        #----------------------------------------------------
        # self.beta_a = np.float64( 1.75 )
        # self.beta_b = np.float64( 1.75 )
        #----------------------------------------------------
        # self.beta_a = np.float64( 1.618034 )
        # self.beta_b = np.float64( 1.618034 ) 
        #----------------------------------------------------
        # self.beta_a = np.float64( 2.2 )
        # self.beta_b = np.float64( 2.2 ) 
        #----------------------------------------------------
        # self.beta_a = np.float64( 1.9 )
        # self.beta_b = np.float64( 1.9 )
        #----------------------------------------------------
        # self.beta_a = np.float64( 1.95 )
        # self.beta_b = np.float64( 1.95 ) 
        #----------------------------------------------------
        # self.beta_a = np.float64( 3.0 )
        # self.beta_b = np.float64( 1.0 )
        #----------------------------------------------------
        # self.beta_a = np.float64( 2.0 )
        # self.beta_b = np.float64( 5.0 )
        #----------------------------------------------------
        # self.beta_a = np.float64( 5.0 )
        # self.beta_b = np.float64( 1.0 )
        #----------------------------------------------------
        # (1/15/13) New parameters to get eta = 1 and then
        # we vary lambda as before to explore threshold.
        # (a < b) => skewed slightly to the left (x=0).
        #----------------------------------------------------
        # self.beta_a = np.float64( 2.0 )
        # self.beta_b = np.float64( 2.63869 )
        # This (a,b) gives eta = 1.0, mu = ???, m = ???        
                                  
        #-----------------------------------         
        # Option to set random number seed
        #-----------------------------------        
        self.seed = None  # (use default/random seed)
        # self.seed = 12675  ##########

        #-------------------------------------       
        # Can override these from __init__()
        #--------------------------------------
        self.SILENT = False
        # self.PLOT   = True
            
    #   read_cfg_file()
    #-------------------------------------------------------------   
    def set_computed_parameters(self):
     
        #-------------------------------------------------
        # Set parameters for the the random fraction, X,
        # that survive a disturbance (Beta distribution)
        #     beta_eta = E[-ln(X)],
        #     beta_m = E(1/X), beta_mu = E(X) 
        #-------------------------------------------------
        ab_sum        = (self.beta_a + self.beta_b)
        self.beta_eta = digamma(ab_sum) - digamma(self.beta_a)
        self.beta_mu  = self.beta_a / ab_sum
        if (self.beta_a != 1):
            self.beta_m   = (ab_sum - 1) / (self.beta_a - 1)
        else:
            self.beta_m   = np.float64(9999999.0)
            print '================================='
            print ' NOTE: m = E(1/X) is infinite.'
            print '       Using 9999999.0 for now.'
            print '================================='
        #-----------------------------------------------
        # If (a < 1) and (b >= (1-a)) the m = Infinity
        #------------------------------------------------       
        if (self.beta_a < 1) and \
           (self.beta_b >= (1 - self.beta_a)):
            self.beta_m   = np.float64(9999999.0)
            print '================================='
            print ' NOTE: m = E(1/X) is infinite.'
            print '       Using 9999999.0 for now.'
            print '================================='


        if (self.exp_lambda > 0):
            #----------------------------------------------
            # Option 1:  Set lambda and compute indicator
            #----------------------------------------------
            self.indicator = self.r - (self.exp_lambda * self.beta_eta)
        else:
            #----------------------------------------------
            # Option 2:  Set indicator and compute lambda
            #----------------------------------------------
            self.exp_lambda = (self.r - self.indicator)/(self.beta_eta)
            
        print 'self.exp_lambda = ', self.exp_lambda
        print 'self.beta_eta   = ', self.beta_eta
        print 'self.indicator  = ', self.indicator
                
        #----------------------------------------
        # Compute the indicators, I1, I2 and I3
        #--------------------------------------------------
        # Note that: r1 <= r2 <= r3, so I_3 <= I_2 <= I1.
        # Also, the values may not be that close, e.g.
        # -1.66 < 0.3 < 0.817
        #--------------------------------------------------
        self.r1 = self.exp_lambda * (1 - self.beta_mu)
        self.I1 = self.r - self.r1
        #----------------------------------------        
        self.r2 = self.exp_lambda * self.beta_eta
        self.I2 = self.r - self.r2
        #-----------------------------------------------
        self.r3 = self.exp_lambda * (self.beta_m - 1)
        self.I3 = self.r - self.r3
        #-----------------------------------------------
        if not(self.SILENT):
            print 'beta_a = ', self.beta_a
            print 'beta_b = ', self.beta_b
            print ' '
            print 'r   = ', self.r
            print 'K   = ', self.K, ' = (r/b)'
            print 'b   = ', self.logistic_b
            print 'lam = ', self.exp_lambda
            print ' '
            print 'r1  = lam * [1 - E(X)]   =', self.r1
            print 'r2  = lam * E[-ln(X)]    =', self.r2
            print 'r3  = lam * [E(1/X) - 1] =', self.r3
            print ' '
            print 'I1  = r - r1 =', self.I1
            print 'I2  = r - r2 =', self.I2
            print 'I3  = r - r3 =', self.I3
            print ' '
            
        #-----------------------------------------------------
        # Set the time parameters, including the parameters
        # for the random intervals between disturbances, now
        # drawn from an Exponential(lambda) distribution.
        #-----------------------------------------------------------------
        # Note: Mean jump size = 1 / lambda. [years]
        #       Mean number of events per year = lambda. [1/year]
        #       lambda -> 0 implies huge jump sizes and no events.
        #       smaller lambda => larger jumps  => fewer events per year
        #       larger  lambda => smaller jumps => more events per year
        #-----------------------------------------------------------------
        # In order to simulate the same total number of events
        # (on average) with a different value of lambda, we need for
        # t_max to be inversely proportional to lambda. e.g.
        #    lambda = 1  (events per year) => t_max = 20
        #    lambda = 10 (events per year) => t_max = 2 [years]
        #-----------------------------------------------------------------
        # mean_t_max = mean_total_n_events * mean_step_size  [years]
        #            = mean_total_n_events * (1 / lambda)
        # We set mean_total_n_events equal to a constant (suff. large)
        #    so we can explore the limit of the expected pop. size.
        #-----------------------------------------------------------------

        #-----------------------------------------------------        
        # Option to simulate the same total number of events
        # (on average) with different values of lambda.
        #  #######  DOUBLE CHECK THIS  ########
        #-----------------------------------------------------
#         self.stop_method = 0
#         self.t_max = (self.mean_total_n_events / self.exp_lambda)  # [years]
#         # self.t_max      = np.float64( 20.0 )   # [years]
#         FS              = 10     # (factor of safety)
#         self.n_seq      = np.int32(self.mean_total_n_events * FS)
        
        #----------------------------------------------------------       
        # Option to simulate a fixed number of disturbance events
        #---------------------------------------------------------- 
        self.stop_method = 1
        self.t_max = -1         #######
        self.n_seq = np.int32( self.mean_total_n_events)
        ## self.n_seq = 1 + np.int32( self.mean_total_n_events)
                
    #   set_computed_parameters()
    #-------------------------------------------------------------
    def get_X_values(self):

        numpy.random.seed( self.seed )
        a = self.beta_a
        b = self.beta_b
        self.X_seq = numpy.random.beta(a, b, size=self.n_seq)

        if not(self.SILENT): 
            print 'min(X_k)      =', self.X_seq.min()
            print 'max(X_k)      =', self.X_seq.max()
            #--------------------------------------------------
            avg1 = np.sum(self.X_seq) / self.n_seq
            avg2 = np.sum(-np.log(self.X_seq)) / self.n_seq
            avg3 = np.sum(1/self.X_seq) / self.n_seq
            #---------------------------------------
            print 'avg(X_k)      =', avg1
            print 'E(X_k)        =', self.beta_mu
            #---------------------------------------
            print 'avg(-ln(X_k)) =', avg2
            print 'E[-ln(X_k)]   =', self.beta_eta 
            #---------------------------------------
            print 'avg(1/X_k)    =', avg3
            print 'E(1/X_k)      =', self.beta_m
            #---------------------------------------           
            print 'n_X           =', self.n_seq
            print ' '
            
    #   get_X_values()
    #-------------------------------------------------------------
    def get_T_values(self):

        #-----------------------------------------------
        # Note: The "scale" parameter required by this
        #       function is the mean or (1/lambda).
        #-----------------------------------------------
        mean = (1 / self.exp_lambda)
        self.T_seq = numpy.random.exponential(scale=mean,
                                              size=self.n_seq)

        if not(self.SILENT):
            print 'min(T_k) =', self.T_seq.min()
            print 'max(T_k) =', self.T_seq.max()
            print ' '
            
    #   get_T_values()
    #-------------------------------------------------------------
#     def get_T_values2(self):
# 
#         # This is not finished.
#         #--------------------------------------------------
#         # Note: This version simulates an inhomogeneous
#         #       Poisson process where lambda increases
#         #       linearly with time from a value that is
#         #       subcritical to one that is supercritical.
#         #--------------------------------------------------
#         # Note: unit_test() sets eta=1, so that I = (r-lam).
#         #       Given I, it computes lam = r - I.
#         #       If we start with I1 = 0.5 (subcritical),
#         #       and let lam2 = (r + 0.5), we'll get
#         #       I2 = (r - lam2) = -0.5 (supercritical).
#         #--------------------------------------------------
#         # Note: The "scale" parameter required by this
#         #       function is the mean or (1/lambda).
#         #--------------------------------------------------
#         I2 = -0.3
#         ## I2   = -0.5
#         ## I2   = -1.0
#         lam1 = self.exp_lambda
#         lam2 = self.logistic_r - I2
#         tmax = self.mean_total_n_events / ((lam1 + lam2)/2)
#         print '### lam1, lam2 = ', lam1, lam2
#         print '### tmax = ', tmax
#         m    = (lam2 - lam1) / tmax
#         #----------------------------------------
#         T_seq    = np.zeros( self.n_seq )
#         mean     = (1 / self.exp_lambda)
#         T1       = numpy.random.exponential( scale=mean )
#         T_seq[0] = T1  # (no copy method, since float)
#         T_sum    = T1  # (no copy method, since float)
#         k        = 1
#         
#         while (k < self.n_seq):
#             lam  = (m * T_sum) + lam1
#             mean = (1 / lam)
#             Tk   = numpy.random.exponential( scale=mean )
#             T_seq[ k ] = Tk
#             T_sum += Tk
#             k     += 1
# 
#         self.T_seq = T_seq
#         
#         if not(self.SILENT):
#             print 'min(T_k) =', self.T_seq.min()
#             print 'max(T_k) =', self.T_seq.max()
#             print ' '
#             
#     #   get_T_values2()
    #-------------------------------------------------------------
    def run_model(self):

        self.initialize()
        while not(self.DONE):
            self.update()
        self.finalize()
        
    #   run_model()
    #-------------------------------------------------------------
    def initialize(self):

        self.read_cfg_file()
        self.set_computed_parameters()   ## (New: 8/19/17)
        self.get_X_values()
        self.get_T_values()
        ## self.get_T_values2()
        
        self.time       = np.float64(0)
        self.jump_index = np.int64(0)
        ### self.jump_index = np.int64(1)

        #---------------------------------------
        # Notice +1 here; need to preserve N_0
        #---------------------------------------         
        self.N    = np.zeros( self.n_seq + 1, dtype='float64' )
        self.N[0] = np.float64( self.N0 )

        #-------------------------------------
        # For integer-valued population size
        #-------------------------------------
        ## self.N    = np.zeros( self.n_seq, dtype='int64' )
        ## self.N[0] = np.int64( self.N0 )

        self.last_N = self.N0.copy()
        ## print 'self.last_N =', self.last_N  ######
        self.DONE = False
        
    #   initialize()
    #-------------------------------------------------------------
    def update(self):

        self.update_T()
        self.update_X()
        self.update_N()
        self.update_time()
            
    #   update()
    #-------------------------------------------------------------
    def update_T(self):
       
        self.T = self.T_seq[ self.jump_index ]
        
    #   update_T()
    #-------------------------------------------------------------
    def update_X(self):
       
        self.X = self.X_seq[ self.jump_index ]
        
    #   update_X()
    #-------------------------------------------------------------
    def update_N(self):

        last_N = self.last_N
        r      = self.logistic_r
        b      = self.logistic_b

        e_term     = np.exp( -r * self.T )
        N_logistic = r / (b + ((r/last_N) - b) * e_term)
        
        #-------------------------------------------------
        # Deterministic growth followed by a disturbance
        # from which a random fraction, X, survive.
        #-------------------------------------------------
        new_N = N_logistic * self.X
        
        #------------------------------------------------------
        # This additional requirement is not currently in the
        # model, and assumes that we need N >= 1. (1/15/13)
        # This is also where we would add Allee effect.
        #------------------------------------------------------
        if (self.N_min > 0):
            if (new_N < self.N_min):
                new_N = 0.0
        # new_N = np.int64( N_logistic * X )   # (force to integer)

        #---------------------------------------
        # Notice +1 here; need to preserve N_0
        #---------------------------------------
        self.N[ self.jump_index + 1 ] = new_N

        # print '(jump_index, time, N) =', self.jump_index, self.time, new_N  ######

        #---------------------------------
        # Do not change value of self.N0
        #---------------------------------
        self.last_N = new_N   #######
    
    #   update_N()
    #-------------------------------------------------------------
    def update_time(self):

        #----------------------------------------------------
        # Notes:  Need to be taking small timesteps during
        #         the deterministic growth so we can create
        #         nice-looking plots.
        #----------------------------------------------------
        self.dt          = self.T
        self.time       += self.dt
        self.jump_index += 1

        if (self.stop_method == 0):
            self.DONE = (self.time > self.t_max)
        else:
            self.DONE = (self.jump_index >= self.n_seq)
        
    #   update_time()
    #-------------------------------------------------------------
    def get_plot_info(self):

        r = self.logistic_r
        b = self.logistic_b
        
        #---------------------------------------------------
        # Get the t and N(t) values for disturbance events
        #---------------------------------------------------
        T_vals     = self.T_seq[:self.jump_index]
        t_jumps = np.cumsum( T_vals )
        #------------------------------------------------
        # Even though N[0]=N0 is not a jump, include it
        #------------------------------------------------
        t_jumps = np.concatenate(([0], t_jumps))
        N_jumps = self.N[:self.jump_index + 1]

        if not(self.SILENT):
            print 'N_jumps ='
            print N_jumps            
            print 't_jumps ='
            print t_jumps
            print ' '
        
        nt    = 20  # (Number of times to plot between events)
        index = 0
        #----------------------------
        self.plot_t_jumps = t_jumps
        self.plot_N_jumps = N_jumps
        #----------------------------
        nj = (t_jumps.size - 1)  ##### (Need the -1 here!)
        self.plot_t_vals = np.zeros(nt * nj, dtype='float64')
        self.plot_N_vals = np.zeros(nt * nj, dtype='float64')

        #----------------------------------------------        
        # We added 0 to t_jumps and N0 to N_jumps, so
        # we need the -1 in the next line.
        #----------------------------------------------
        for k in xrange( t_jumps.size - 1 ):
            T_k    = self.T_seq[k]
            #-----------------------------------------------------
            # Note that linspace values include 0 and T_k itself.
            # Also, recall a[i: i+5] has 5 values in Python.
            #-----------------------------------------------------
            t_vals = np.linspace(0, T_k, nt)  # (start back at 0)
            self.plot_t_vals[ index: index + nt ] = (t_vals + t_jumps[k])
            #---------------------------------------------
            N0     = self.N[k]   # (get the next "N0")
            e_term = np.exp( -r * t_vals)
            N_vals = r / (b + ((r/N0) - b) * e_term)
            self.plot_N_vals[ index: index + nt ] = N_vals
            #---------------------------------------------
            index += nt

        #----------------------------------------------
        # Include very last "jump point" for plotting
        # Note: Use final_N; name last_N is taken.
        #----------------------------------------------
        final_t = [ t_jumps[-1] ]
        final_N = [ N_jumps[-1] ]     
        self.plot_t_vals = np.concatenate((self.plot_t_vals, final_t))
        self.plot_N_vals = np.concatenate((self.plot_N_vals, final_N))
        
#         print 'self.N[0] =', self.N[0]
#         print '-------------------------------------------'        
#         print 'plot_t_vals.shape =', self.plot_t_vals.shape
#         print 'plot_N_vals.shape =', self.plot_N_vals.shape
#         print '-------------------------------------------'
#         print 'plot_t_jumps.shape =', self.plot_t_jumps.shape
#         print 'plot_N_jumps.shape =', self.plot_N_jumps.shape
        
    #   get_plot_info()
    #-------------------------------------------------------------
    def plot_N_vs_t(self):

        self.get_plot_info()
        
        label_size = 15
        marker_size = 3
        
        #----------------------------------
        # Method 1. Use matplotlib.pyplot
        #----------------------------------
        fig   = pyplot.figure()
        ax    = fig.add_subplot(111)
        c1 = ax.plot( self.plot_t_vals, self.plot_N_vals )
        # c2 = ax.plot( self.plot_t_vals, self.plot_N_vals, 'r+' )
        c3 = ax.plot( self.plot_t_jumps, self.plot_N_jumps, 'ro',
                      markersize=marker_size)
        pyplot.setp(ax.get_xticklabels(), fontsize=label_size)
        pyplot.setp(ax.get_yticklabels(), fontsize=label_size)
        title = 'Randomly Disturbed Logistic Growth'
        title += ' (r =' + str(self.logistic_r) + ')'
        ax.set_title( title, fontsize=label_size)
        ax.set_xlabel( 'time [years]', fontsize=label_size )
        ax.set_ylabel( 'Population size, N(t)', fontsize=label_size )
        pyplot.show()
            
    #   plot_N_vs_t()
    #-------------------------------------------------------------
    def finalize(self):

        if not(self.SILENT):
            print 'Finished with simulation.'
            print 'Final time             =', self.time, ' [years]'
            print 'Number of disturbances =', self.jump_index
            print 'Final population size  =', self.N[ self.jump_index - 1 ]
            print ' '
    ##        print 'N-values ='
    ##        print self.N[: self.jump_index]
    ##        print ' '

        if (self.PLOT):
            self.plot_N_vs_t()

    #   finalize()
#-----------------------------------------------------------------   
#-----------------------------------------------------------------        
def monte_carlo(n_reps=50, mean_total_n_events=30, eps=0.1,
                r_min=0.2, r_max=2.2, n_r=11,  # (steps of 0.2)
                m_min=1.1, m_max=4.0, n_m=60,
                lam_min=0.1, lam_max=6.0, n_lam=60):

    ## n_dt = 100  # (only used to size "results"; not used now)
    PRINT_TO_FILE = True   ######
    
    # (Make this a class ??)
    # (add ability to set t_max here.)
    start_time = time.time()
    
    print 'Parameter ranges for logistic model:'
    print '   n_reps =', n_reps
    print '   r   in [' + str(r_min) + ', ' + str(r_max) + '], n_r = ' + str(n_r)
    print '   m   in [' + str(m_min) + ', ' + str(m_max) + '], n_m = ' + str(n_m)
    print '   lam in [' + str(lam_min) + ', ' + str(lam_max) + '], n_lam = ' + str(n_lam)
    print 'Starting simulations...'
    
    #-----------------
    # Set parameters
    #-----------------
    eps     = np.float64( eps )
    #--------------------------------------
    r_min   = np.float64( r_min )
    r_max   = np.float64( r_max )
    n_r     = np.int32( n_r )
    #--------------------------------------
    m_min   = np.float64( m_min )
    m_max   = np.float64( m_max )
    n_m     = np.int32( n_m )
    #--------------------------------------
    lam_min = np.float64( lam_min )
    lam_max = np.float64( lam_max )
    n_lam   = np.int32( n_lam )
    #--------------------------------------
    dm    = (m_max - m_min) / n_m
    m_vec = np.linspace(m_min, m_max, n_m)  ################
    ## m_vec = m_min + np.arange(m_min, m_max, dm)
    ## m_vec = np.linspace(dm, m_max, n_m)
    #-----------------------------------------------------    
    beta_a_min  = (m_max - 1 - eps) / (m_max - 2)
    beta_b      = beta_a_min - eps
    beta_a_vec  = 1 + ((beta_a_min - eps)/(m_vec - 1))  # (descends from max to min)
    #-----------------------------------------------------
    lam_vec = np.linspace(lam_min, lam_max, n_lam)
    #-----------------------------------------------------
    r_vec   = np.linspace(r_min, r_max, n_r)
    #-----------------------------------------------------
    n_sims    = (n_r * n_lam * n_m * n_reps)
    sim_count = np.int64(0)
    last_time = start_time
    
    #------------------------------------
    # Create big array to store results
    #------------------------------------
    ## results = np.zeros([n_r, n_lam, n_m, n_reps, n_dt], dtype='float64')
    means   = np.zeros([n_r, n_lam, n_m], dtype='float64')
    
    #---------------------------
    # Create instance of model
    #---------------------------
    model = logistic_model()
    model.SILENT = True
    model.PLOT   = False
    model.beta_b = beta_b
    model.mean_total_n_events = mean_total_n_events
    
    #-------------------------------------------------------
    # Perform n_reps runs for each set of parameter values
    #-------------------------------------------------------
    for r_index in xrange(n_r):
        model.logistic_r = r_vec[ r_index ]
        for lam_index in xrange(n_lam):
            model.exp_lambda = lam_vec[ lam_index ]
            for m_index in xrange(n_m):
                model.beta_a = beta_a_vec[ m_index ]
                Nt_sum = np.float64( 0 )
                for n in xrange(n_reps):
                    model.run_model()
                    n_jumps = model.jump_index - 1  #######  DOUBLE_CHECK
                    Nt = model.N[ n_jumps ]
                    Nt_sum += Nt
                    ## results[ r_index, lam_index, m_index, n, :n_jumps ] = model.N[:n_jumps]
                    sim_count += 1
                    #-------------------------
                    # Check the elapsed time
                    #-------------------------
                    elapsed_time = (time.time() - last_time)  # [seconds]
                    if (elapsed_time > 5):
                        percent = 100 * np.float32(sim_count) / n_sims
                        msg1 = '  count = ' + str(sim_count) + ' of ' + str(n_sims) 
                        print msg1, '(', "%5.1f" % percent, '%)'
##                        print '  lambda  =', lam_vec[ lam_index ]
##                        print '  n_jumps =', n_jumps
##                        print '  time    =', model.time
##                        print ' '
                        ## print 'count =', sim_count, 'of', n_sims, '(=', int(percent), '%)'
                        last_time = time.time()
                    
                #------------------------------------------------------
                # Compute statistics restricted to positive N-values.
                #------------------------------------------------------
                Nt_mean = (Nt_sum / n_reps)
                # Nt_mean = (np.sum(results[r_index, m_index, lam_index, :]) / n_reps)
                means[ r_index, lam_index, m_index ] = Nt_mean
                #--------------
                # For testing
                #--------------
##                print 'r       =', r_vec[ r_index ]
##                print 'm       =', m_vec[ m_index ]
##                print 'lam     =', lam_vec[ lam_index ]
##                print 'Nt_mean =', Nt_mean
##                print '-----------------------------------'

    #-----------------
    # Print a report
    #-----------------
    run_time_sec = (time.time() - start_time)
    run_time_min = (run_time_sec / 60.0)
    print 'Run time =', run_time_min, ' [minutes]'
    print ' '

    #----------------------------
    # Print data to a text file
    #----------------------------
    if (PRINT_TO_FILE):
        out_file = 'Logistic_Output1.txt'
        out_unit = open( out_file, 'w' )
        #--------------------
        # Write header info
        #--------------------
        out_unit.write('Parameter ranges for logistic model:\n\n')
        out_unit.write('n_reps = ' + str(n_reps) + '\n')
        out_unit.write('r   in [' + str(r_min) + ', ' + str(r_max) + '], n_r = ' + str(n_r) + '\n')
        out_unit.write('m   in [' + str(m_min) + ', ' + str(m_max) + '], n_m = ' + str(n_m) + '\n')
        out_unit.write('lam in [' + str(lam_min) + ', ' + str(lam_max) + '], n_lam = ' + str(n_lam) + '\n')
        out_unit.write('\n')
        out_unit.write( 'E[N_n](m, lam)\n' )
        ## out_unit.write( 'X-axis: m = E(1/X)\n' )
        out_unit.write( 'X-axis: m = E[-ln(X)]\n' )
        out_unit.write( 'Y-axis: lambda\n')
        out_unit.write( '\n' )
                       
        #-----------------------------------
        # Write "means" array to text file
        #-----------------------------------
        for r_index in xrange(n_r):
            out_unit.write('r = ' + str(r_vec[r_index]) + '\n' )
            for lam_index in xrange(n_lam):
                means[r_index, lam_index, :].tofile( out_unit, sep=" " )
                out_unit.write( '\n' )
            out_unit.write( '\n' )
        #-----------------
        # Close the file
        #-----------------
        out_unit.close()
        print 'Finished writing data to ' + out_file + '.'
        print ' '
        
    #-----------------------------------------------
    # Make a color density plot for first r-value
    # Recall numpy arrays have form: (z, y, x),
    #   so m will be on the x_axis, lam on y-axis.
    #-----------------------------------------------
    data  = means[ 0, :, : ]
    dmin  = data.min()
    dmax  = data.max()
    print 'min( E[N_n] ) =', dmin
    print 'max( E[N_n] ) =', dmax
    print ' '
    
    #----------------------------------
    # Method 1. Use matplotlib.pyplot
    #----------------------------------
    fig   = pyplot.figure()
    ax    = fig.add_subplot(111)
    cax   = ax.imshow( data, cmap=cmaps.jet, origin='lower',
                       extent=(m_min, m_max, lam_min, lam_max),
                       aspect='auto',  # (don't use 'equal' here)
                       interpolation='nearest' )
    r_str = str( r_vec[0] )
    ax.set_title('E[N_n](m, lam) for r=' + r_str)
    ## ax.set_xlabel('m = E(1/X)')
    ax.set_xlabel('m = E[-ln(X)]')
    ax.set_ylabel('lambda')
##    ax.set_xlim( 0, m_max )  # (affects image aspect ratio, etc.)
##    ax.set_ylim( 0, lam_max )

    #-----------------   
    # Add a colorbar
    #-----------------
    N_max = model.N_max
    cbar = fig.colorbar( cax, ticks = np.linspace(0,N_max,10) )
    # cbar.ax.set_yticklabels([
##    pyplot.xlim(0, m_max)     # (affects image aspect ratio, etc.)
##    pyplot.ylim(0, lam_max)
    pyplot.show()

    #----------------------------------
    # Method 2. Use pylab.pcolor
    #----------------------------------

#   monte_carlo()
#-----------------------------------------------------------------
##def monte_carlo_print_to_file():
##
##    #--------------------------
##    # Open text file to write
##    #--------------------------
##    out_file = 'Logistic_Output1.txt'
##    out_unit = open( out_file, 'w' )
##    
##    #--------------------
##    # Write header info
##    #--------------------
##    out_unit.write('Parameter ranges for logistic model:\n\n')
##    out_unit.write('n_reps = ' + str(n_reps) + '\n')
##    out_unit.write('r   in [' + str(r_min) + ', ' + str(r_max) + '], n_r = ' + str(n_r) + '\n')
##    out_unit.write('m   in [' + str(m_min) + ', ' + str(m_max) + '], n_m = ' + str(n_m) + '\n')
##    out_unit.write('lam in [' + str(lam_min) + ', ' + str(lam_max) + '], n_lam = ' + str(n_lam) + '\n')
##    out_unit.write('\n')
##    out_unit.write( 'E[N_n](m, lam)\n' )
##    out_unit.write( 'X-axis: m = E(1/X)\n' )
##    out_unit.write( 'Y-axis: lambda\n')
##    out_unit.write( '\n' )
##                   
##    #-----------------------------------
##    # Write "means" array to text file
##    #-----------------------------------
##    for r_index in xrange(n_r):
##        out_unit.write('r = ' + str(r_vec[r_index]) + '\n' )
##        for lam_index in xrange(n_lam):
##            means[r_index, lam_index, :].tofile( out_unit, sep=" " )
##            out_unit.write( '\n' )
##        out_unit.write( '\n' )
##    #-----------------
##    # Close the file
##    #-----------------
##    out_unit.close()
##    print 'Finished writing data to ' + out_file + '.'
##    print ' '

#   monte_carlo_print_to_file()
#-----------------------------------------------------------------
##def monte_carlo_plot_lambda_vs_mu():
##
##    #-----------------------------------------------
##    # Make a color density plot for first r-value
##    # Recall numpy arrays have form: (z, y, x),
##    #   so m will be on the x_axis, lam on y-axis.
##    #-----------------------------------------------
##    data  = means[ 0, :, : ]
##    dmin  = data.min()
##    dmax  = data.max()
##    print 'min( E[N_n] ) =', dmin
##    print 'max( E[N_n] ) =', dmax
##    print ' '
##    
##    #----------------------------------
##    # Method 1. Use matplotlib.pyplot
##    #----------------------------------
##    fig   = pyplot.figure()
##    ax    = fig.add_subplot(111)
##    cax   = ax.imshow( data, cmap=cmaps.jet, origin='lower',
##                       extent=(m_min, m_max, lam_min, lam_max),
##                       aspect='auto',  # (don't use 'equal' here)
##                       interpolation='nearest' )
##    r_str = str( r_vec[0] )
##    ax.set_title('E[N_n](m, lam) for r=' + r_str)
##    ax.set_xlabel('m = E(1/X)')
##    ax.set_ylabel('lambda')
####    ax.set_xlim( 0, m_max )  # (affects image aspect ratio, etc.)
####    ax.set_ylim( 0, lam_max )
##
##    #-----------------   
##    # Add a colorbar
##    #-----------------
##    N_max = model.N_max
##    cbar = fig.colorbar( cax, ticks = np.linspace(0,N_max,10) )
##    # cbar.ax.set_yticklabels([
####    pyplot.xlim(0, m_max)     # (affects image aspect ratio, etc.)
####    pyplot.ylim(0, lam_max)
##    pyplot.show()
##
##    #----------------------------------
##    # Method 2. Use pylab.pcolor
##    #----------------------------------

#   monte_carlo_plot_lambda_vs_mu()
#-----------------------------------------------------------------
def plot_data_from_file(in_file='Logistic_Output1.txt',
                        frame=1, nx=60, ny=60):

    if (frame > 11) or (frame < 1):
        print 'ERROR: frame number must be in [1,11].'
        print ' '
        return
    
    n_header = 11
    m_min    = 1.1
    m_max    = 4.0
    lam_min  = 0.1
    lam_max  = 6.0
    
    in_unit = open(in_file, 'r')
    os.chdir( '/Users/peckhams/Desktop/')
    
    #------------------------
    # Read the header lines
    #------------------------
    for k in xrange(n_header):
        line = in_unit.readline()

    #------------------------------------
    # Read down to the specified "frame"
    #------------------------------------
    for j in xrange(frame):
        r_line = in_unit.readline()
        r_str = r_line[-4:]  ######
        data = np.fromfile( in_unit, dtype='float64',
                               sep=" ", count=(nx*ny) )

    data = np.reshape(data, (ny, nx))
    print 'shape =', np.shape(data)
    
    #-----------------
    # Close the file
    #-----------------
    in_unit.close()

    #------------------
    # Create the plot
    #------------------
    fig   = pyplot.figure()
    ax    = fig.add_subplot(111)
    cax   = ax.imshow( data, cmap=cmaps.jet, origin='lower',
                       extent=(m_min, m_max, lam_min, lam_max),
                       aspect='auto',  # (don't use 'equal' here)
                       interpolation='nearest' )
    ax.set_title('E[N_n](m, lam) for r=' + r_str)
    ## ax.set_xlabel('m = E(1/X)')
    ax.set_xlabel('m = E[-ln(X)]')
    ax.set_ylabel('lambda')
    print 'Finished with plot for frame =', frame, '.'
    print ' '
    
    #-----------------   
    # Add a colorbar
    #-----------------
    ## N_max = model.N_max
    d_max = data.max()
    cbar = fig.colorbar( cax, ticks = np.linspace(0,d_max,10) )
    pyplot.show()
   
#   plot_data_from_file()
#-------------------------------------------------------------
    
    


    


