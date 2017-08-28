# pyDirectionCompSense

Compressed sensing for directional data. This dataset will be written from scratch, attempting to do so in Cython, so that we can have a code that is optimal for data that is directional in nature (e.g. diffusion data). By doing this, we have a chance at creating our own method for solving the compressed sensing minimization problem.

The idea is that we will write the code in a modular fashion, looking for where to optimize when we need to. As per Jason Lerch, "The first rule of optimization... Don't", implying that we should only optimize when it's absolutely necessary. 

The code will first be written in Python and then profiled -- all of this being noted in the commits (hopefully), in order to get things up and running. The plan is to:

  <b> Proof of Principle
  
    1) Have the code be able to properly perform TV operations on single slice (2D) datasets
    
    2) Have the code properly run FFT's (including knowledge of all normalization factors) as well as XFMs (Wavelet), and calculate the gradients for them
    
    3) Be able to feed this information in, along with pseudo-undersampled data of high SNR phantoms into an optimization algorithm (scipy.minimize to start) utilizing CS reconstruction techniques and compare to the gold standard (fully sampled) case
  
  <b> Secondary Testing 
  
    1) Build the foundation of the directional term in two ways:
    
      a. Using the method of a weighting term that is calculated via some f(dot(di,dj)) -- most likely a Gaussian term
      
      b. Using the method of an LSQ fitting technique with the images around it, using A as a matrix of the differences of the vector directions.
    
    2) Test the Diffusion phantom data without the directional terms
    
    3) Test the Diffusion phantom data with the directional terms and compare to each other, without the term, and the gold standard (fully sampled case)
  
  <b> Tertiary Testing
  
    1) Actually undersample data from the scanner
    
    2) Test reconstruction on this data -- both with and without the directional term (depending on the data that we're working with, if it's dMRI or just plain MRI)
    
    
