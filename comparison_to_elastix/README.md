### Quick comparison to Elastix for the Mouse Nissl example

The images below show a cross-section of the "fixed" and "moving" brains that were registered with LDDMM, and Elastix (v. 4.900).  
Green-Magenta composites show the overlay of the fixed image and the now _deformed_ moving brain. Ideal overlap should yield gray.<br/>
 However, that is generally only possible for identical images, and what we are looking is a "blend" between green and magenta.<br/> Where we see only green or only magenta, there is no overlap (like parts of the olfactory bulb)).

![fixed_and_moving](assets/README-93549476.png)
Fixed an moving image before deformation.
![composite_comparison_a](assets/README-d37b1745.png)
Composite A
![composite_comparison_b](assets/README-6b74997b.png)
Composite B
![composite_comparison_c](assets/README-c740d282.png)
Composite C


I used 400 iterations, but otherwise default parameters for LDDMM Registration
```python
p = 2
sigmaM = 10.0
eL = 5e-7
eT = 2e-5
eV = 5e-4
naffine = 50
niter = 400
sigmaR = 2e-1
a = (xI[0][1]-xI[0][0])*2
```

Note that I converted the .hdr files to .tiffs. I had some issues with bSpline which
I think was due to the *FinalGridSpacingInVoxels* and *SampleRegionSize* in the
bSpline parameter file. Parameter files are attached.

I used `PMD3097_orig_target_STS_clean.hdr` as the _fixed_ image and `PMD2052_orig_target_STS_clean.hdr` as the _moving_ image.

Interestingly, LDDMM performs really well compared to Elastix, however, Elastix does seem to match smaller structures somewhat better (see hippocampus) and does not artificially extend the olfactory bulb. Despite this, LDDMM looks promising, and I am sure the author could adjust the parameters so as to make the
registration work better.
