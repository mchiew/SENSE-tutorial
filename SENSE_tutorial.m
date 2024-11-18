%% SENSE Parallel Imaging
% Mark Chiew (mark.chiew@utoronto.ca)
% 
% This MATLAB tutorial gives an introduction to SENSE parallel imaging in 
% MRI. It walks through the estimation of coil sensitivities, combining images 
% from multiple coils, and reconstruction of under-sampled data using the SENSE 
% algorithm.
% 
% The data required for this can be downloaded from:
% 
% https://github.com/mchiew/SENSE-tutorial/blob/main/raw_data.mat
%
% Changelog
% 2014: Initial version
% 2022: Migrated to Github
% 2024: Updated email, raw data link

%% Load data
%%
load('raw_data.mat');
%% Explore and view sensitivities
% There is a single raw 4D array containing the data we'll be working with, 
% named |raw|

[Nx,Ny,Nz,Nc] = size(raw)
%% 
% The dimensions of the dataset are |(Nx, Ny, Nz, Nc)| where |(Nx, Ny, Nz)| 
% correspond to the spatial image matrix size, and |Nc| corresponds to the number 
% of difference receiver coils ( or coil channels).
% 
% In this case, we're working with a |96x96| 2D image |(Nz=1)|, that has 
% |16| different coils. 
% 
% The raw data are provided exactly as measured - which means this is complex 
% k-space data. We get one k-space dataset per coil, which we can view the magnitude 
% of:

show_grid(log(abs(raw)), [-2 8], jet)
%% 
% How do the k-space data for each coil look similar/different?
% 
% The coil in the botton right (coil |#16|), for example, and the one right 
% above it (coil |#15|) look very similar in k-space magnitude. Do we expect them 
% to look the same in image space?
% 
% To look at the images, we need to transfom the k-space data into image 
% space, via the inverse discrete Fourier Transform (DFT). The |FFT| (and its 
% inverse |iFFT|) are the most efficient ways to compute DFTs. We have defined 
% some helper functions to do this properly for us - they just make sure that 
% our imaging conventions match up with MATLAB's FFT conventions:

img = ifft2c(raw);    
show_grid(abs(img),[0 16], gray);
%% 
% * How do the images look different?
% * What do you think this tells you about the physical coil array?
% * Do you notice anything about the k-space magnitude data that tells you about 
% the coil images?
%% Combine images
% Before we talk more about coil sensitivities, lets consider the simple problem 
% of combining our multiple coil images into a single representative brain image.
% 
% That means we want to define a transform that takes our |[Nx, Ny, 1, Nc]| 
% data and maps it to a single |[Nx, Ny, 1, 1]| 2D image. For example, we could 
% just take the mean:

img_combined = mean(img,4);
show_img(abs(img_combined),[]);
%% 
% This image is clearly not an accurate representation of what we expect 
% this brain slice to look like. What we would like a coil combination to produce 
% is an image that does not actually have any coil-specific weighting or influence 
% - we would to see what the brain looks like, not what the coils look like.
% 
% * What are the problems here, and why do they occur?
% * Can you come up with a better coil-combination transform? 
% 
% Hint: Try transforms of the form: 
% 
% $${\left(\sum_{l=1}^{N_c } {\left|z\right|}^n \right)}^{\frac{1}{n}}$$

% define and view your own coil-combine transforms
% [Nx, Ny, 1, Nc] -> [Nx, Ny] 
% Try linear and non-linear operations!
% e.g.:
img_combined  = sqrt(sum(abs(img).^2,4));
show_img(abs(img_combined),[]);
%% Estimate sensitivities
% We know that we can model the coil images using a continuous representation 
% with the following equation:
% 
% $C_l \left(x\right)=S_l \left(x\right)\cdot M\left(x\right)$        [1]
% 
% where $C_l$ is image from coil $l$, $S_l$ is the _sensitivity_ of coil 
% $l$, and $M$is the underlying sample magnetisation. The $\left(\cdot \right)$ 
% operator denotes point-wise multiplication here.
% 
% We can write this more explicitly in matrix form, after discretisation 
% as:
% 
% $c=S\;m$         [2]
% 
% by vertically concatenating across the coil dimension, to fit everything 
% into a single equation. Now $m$ is an $N_x N_y \times 1$ vectorised image, $S$ 
% is a vertical concatenation of $N_c$ diagonal matrices such that the diagonal 
% entries of the $l^{\mathrm{th}}$ matrix are $N_x N_y \times 1$ values of $S_l$. 
% This makes $S$ a $N_x N_y N_c \times N_x N_y$ matrix, which means $c$ has dimensions 
% $N_x N_y N_c \times 1$, which is consistent with having $N_c$ images of size 
% $N_x N_y$. An explicit example of a case where $N_x =N_y =N_c =2$:
% 
% $$\left\lbrack \begin{array}{c}c_{11}^1 \\c_{12}^1 \\c_{21}^1 \\c_{22}^1 
% \\c_{11}^2 \\c_{12}^2 \\c_{21}^2 \\c_{22}^2 \end{array}\right\rbrack =\left\lbrack 
% \begin{array}{cccc}s_{11}^1  & 0 & 0 & 0\\0 & s_{12}^1  & 0 & 0\\0 & 0 & s_{21}^1  
% & 0\\0 & 0 & 0 & s_{22}^1 \\s_{11}^2  & 0 & 0 & 0\\0 & s_{12}^2  & 0 & 0\\0 
% & 0 & s_{21}^2  & 0\\0 & 0 & 0 & s_{22}^2 \end{array}\right\rbrack \left\lbrack 
% \begin{array}{c}m_{11} \\m_{12} \\m_{21} \\m_{21} \end{array}\right\rbrack$$
% 
% The only thing you have to work with is $c$. What we need to do however, 
% is estimate $S$, despite the fact that both $S$ and $m$ are unknown. Unfortunately, 
% because the RHS of equation  [2] has more unknowns than there are knowns on 
% the LHS, this problem has no solution.
% 
% Sometimes, an additional image is taken with what is called a _body coil, 
% _which can be used to approximate $m$ but the body coil image still carries 
% its own bias, and it is not available on all systems, and it requires extra 
% time. Instead, we can try and estimate $m$ directly ourselves, from the data 
% we already have.
% 
% Try using your coil-combined image from the previous section as an approximation 
% of $m$. Then solve for $S$ by dividing out $m$ from $c$:

S_0 = img./img_combined;
%S_0 = bsxfun(@rdivide, img, img_combined); % for older platforms
show_grid(abs(S_0),[0 1],jet);
%% 
% You should hopefully see something like smoothly varying images that reflect 
% the overall shading you saw in the raw coil images, but without any of the actual 
% brain structure or contrast!
% 
% [N.B If you don't see this, you should probably go back and try to refine 
% your combined coil image before you continue]
% 
% It turns out that we have an extra piece of information we can use to solve 
% our estimation problem - coil sensitivities must be _smooth_ over space, in 
% all directions. This is because coil sensitivities can be related to the static 
% magnetic fields produced by the same coils with some applied direct current, 
% which you can calculate using the coil geometry and the Biot-Savart law. However, 
% it is enough to know that because of this, the coil sensitivities we estimate 
% must be smooth, and we can use this constraint to help regularise or solve our 
% problem! 
% 
% Try and refine your sensitivity estimates by enforcing a smoothness constraint:
% 
% Hint: The easiest thing to do here is something like convolving your estimate 
% with a smoothing kernel. Alternatively, you can try fitting your data to some 
% low-order polynomial model, or restricting the spatial frequencies of the input 
% data by spatial filtering, or penalising the spatial variation in your estimate 
% and solving your own optimisation problem

% Define your own S_1 which is a smoothed version of S_0
% e.g.:
kernel = ones(9)/9^2;
S_1 = zeros(Nx,Ny,Nz,Nc);
for i = 1:16
    S_1(:,:,1,i) = conv2(S_0(:,:,1,i),kernel,'same');
end
show_grid(abs(S_1),[0 1],jet);
% polynomial fitting

%% 
% The last step here is to mask the sensitivities. Because they won't be 
% well defined anywhere outside of the brain (because $m=0$ in those areas), we 
% should mask the sensitivities to retain only the well-defined regions. We won't 
% lose anything here, because we don't strictly _need_ to know what the sensitivity 
% is outside the brain anyway, so we can safely set those regions to 0.
% 
% *Hint*: Try and find some threshold or some other classifier that distinguishes 
% brain vs. background and define a brain mask

% Define your own S_2 which is a masked version of S_1
% e.g.:
thresh = 0.05*max(abs(img_combined(:)));
mask = abs(img_combined) > thresh;
S_2 = S_1.*mask;
show_grid(abs(S_2),[0 1],jet);
%% 
% For reference, here are sensitivities estimated via a method published 
% in (Walsh et al., _Magn Reson Med _2000):

S_ref = adaptive_est_sens(img);
% Your S_2 on the left, the reference on the right
show_grid(cat(2,abs(S_2),abs(S_ref)),[0,1],jet);
%% Use sensivities 
% Now that we have the sensitivities (feel free to use your own or the reference 
% sensitivities), we can use them to actually solve the "coil-combine" problem 
% in a true least-squares sense, rather than the ad-hoc approach we took above.
% 
% Referring back to the linear equation [2], with both $c$ and $S$ known, 
% it should be straightforward to come up with the _least-squares optimal_ |[Nx, 
% Ny, 1, Nc] to| |[Nx, Ny, 1, 1]| transform to apply to the $N_c$ images in $c$, 
% to recover the single image $m$. 
% 
% *Hint*: It might be easiest to work it out on paper first from Eq. 2, analytically, 
% before figuring out how to implement it 

% Define your new coil-combine transform using
% your knowlwedge of S
img_combined_opt = sum(img.*conj(S_2),4)./sum(S_2.*conj(S_2),4);
show_img(abs(img_combined_opt),[0,48],gray);
%% SENSE Parallel Imaging
% Now, the utility of multi-channel array coils really comes from "parallel 
% imaging", which is a term used to denote reconstruction of images from under-sampled 
% data using coil sensitivity information. The _parallel_ in parallel imaging 
% refers to the fact that each coil channel measures it's own version of the image 
% in parallel with all the others, so there's no time cost to additional measurements. 
% The number of parallel channels is limited by hardware and coil array design 
% considerations.
% 
% SENSE is one of the earliest formulations of parallel imaging, and stands 
% for SENSitivity Encoding (Pruessmann et al., _Magn Reson Med _1999). It formulates 
% the problem and its solution in the contexts of the same linear equations and 
% least-squares solutions we considered in the previous section.
% 
% First, lets consider what happens to our images when we "under-sample" 
% our k-space data. To do this, we can zero/mask out the k-space values we want 
% to pretend to not have acquired:

raw_R2 = raw;
raw_R2(2:2:end,:,:,:) = 0;
img_R2 = ifft2c(raw_R2);
show_grid(abs(img_R2),[0 16],gray)
%% 
% Collecting only half of the necessary k-space data in this way results 
% in _aliasing_ of the brain image, so that we can no longer unambiguously separate 
% out the top and bottom portions of the brain. This is unsurprising, because 
% we've thrown out half of our measurements!
% 
% Lets model our under-sampled imaging problem:
% 
% $k=\Phi \;F\;S\;m$        [3]
% 
% where $S$ and $m$ are defined as in Eq. 2, $F$is the DFT, and $\Phi$ is 
% a diagonal matrix with only |1|s and |0|s on the diagonal, indicating which 
% data has been sampled (|1|) or not (|0|).
% 
% When all the data is sampled, $\Phi$ is the identity matrix, and so we 
% can see Eq. 3 is entirely consistent with Eq. 2:
% 
% $c=F^{-1} k=S\;m$         [4]
% 
% To solve for $m$ from Eq. 3, in the least-squares sense, we get:
% 
% $\hat{m} ={\left.{\left(\left(\Phi \;F\;S\right)\right.}^* \left(\Phi \;F\;S\right)\right)}^{-1} 
% {\left(\Phi \;F\;S\right)}^* k={\left(S^* F^{-1} \Phi \;F\;S\right)}^{-1} S^* 
% F^{-1} \Phi \;k$       [5]
% 
% when you note that $\Phi^* \Phi =\Phi$. However, representing this inverse 
% explicitly is very costly, both in terms of memory and computation time.
% 
% * How much memory would it cost you to solve Eq. 5 directly?
% 
% Luckily, when the matrix $\Phi$ takes on specific forms, the aliasing patterns 
% result in many smaller sub-problems that can be solved easily and independently. 
% Specifically, when $\Phi$ samples every $R^{\mathrm{th}}$ line of k-space, we 
% get regular aliasing (overlap) patterns, so that any given voxel overlaps with 
% at most $R-1$ other voxels. This turns a $N_x N_y$-dimensional problem into 
% $\frac{N_x N_y }{R}$ $R$-dimensional subproblems.
% 
% Notice, for example, in the images plotted above, that within the field-of-view, 
% each coil image is the superposition (alias or overlap) of the "true" image, 
% plus a copy shifted by $\frac{N_x }{2}$ in the "up-down" or x-direction. So 
% we can model each subproblem as:
% 
% $\left\lbrack \begin{array}{c}c_{x,y}^1 \\c_{x,y}^2 \\\vdots \\c_{x,y}^{N_c 
% } \end{array}\right\rbrack =\left\lbrack \begin{array}{cc}S_{x,y}^1  & S_{x+N_x 
% /2,y}^1 \\S_{x,y}^2  & S_{x+N_x /2,y}^2 \\\vdots  & \vdots \\S_{x,y}^{N_c }  
% & S_{x+N_x /2,y}^{N_c } \end{array}\right\rbrack \left\lbrack \begin{array}{c}m_{x,y} 
% \\m_{x+N_x /2,y} \end{array}\right\rbrack$      [6]
% 
% So, as we solve the unaliasing problems for the top half of the aliased 
% images, we get solutions for "true" top voxel value, as well as the aliased 
% voxel from $N_x /2$ below. 
% 
% To unalias your image, you need to solve Eq. 6. The most common and natural 
% way to do this is via least-squares. Implementing a least-squares solver for 
% Eq. 6 might look something like this:

% initialise output image
img_R2_SENSE = zeros(Nx,Ny);
% loop over the top-half of the image
for x = 1:Nx/2
    % loop over the entire left-right extent
    for y = 1:Ny
        % pick out the sub-problem sensitivities
        S_R2 = transpose(reshape(S_2([x x+Nx/2],y,1,:),2,[]));
        % solve the sub-problem in the least-squares sense
        img_R2_SENSE([x x+Nx/2],y) = pinv(S_R2)*reshape(img_R2(x,y,1,:),[],1);
    end
end
% plot the result
show_img(abs(img_R2_SENSE),[0 32],gray);
%% 
% Hopefully, this looks like a normal brain image, despite the fact that 
% half the information in k-space is missing! The SENSE method takes sensitivities 
% and aliased coil images as input, and outputs a single unaliased image, which 
% is what we get here.
% 
% Try solving the |R=3| problem:

raw_R3 = raw;
raw_R3(2:3:end,:,:,:) = 0;
raw_R3(3:3:end,:,:,:) = 0;
img_R3 = ifft2c(raw_R3);
show_grid(abs(img_R3),[0 16],gray)
%% 
% 

% define your solution to the R=3 problem
img_R3_SENSE = SENSE(img_R3,S_2,3);
show_img(abs(img_R3_SENSE),[0 16],gray);
%% 
% and the |R=4| subproblem:

raw_R4 = raw;
raw_R4(2:4:end,:,:,:) = 0;
raw_R4(3:4:end,:,:,:) = 0;
raw_R4(4:4:end,:,:,:) = 0;
img_R4 = ifft2c(raw_R4);
show_grid(abs(img_R4),[0 16],gray)
% define your solution to the R=4 problem
img_R4_SENSE = SENSE(img_R4,S_2,4);
show_img(abs(img_R4_SENSE),[0 16],gray);
%% 
% Try performing the reconstruction using your less refined sensitivity 
% estimates |S_0| and |S_1|, and the reference sensitivities |S_ref|. How do the 
% reconstructions differ?

% R=4 SENSE reconstruction with S_0, S_1, and S_ref
img_S0 = SENSE(img_R4,S_0,4);
img_S1 = SENSE(img_R4,S_1,4);
img_SREF = SENSE(img_R4,S_ref,4);
show_grid(abs(cat(4,img_S0,img_S1,img_R4_SENSE,img_SREF)),[0 16],gray)
%% 
% Although the reconstruction using |S_0| looks best here, in reality the 
% |S_0| sensitivities are over-fit to this data, and represents an unrealistic 
% case since we typically wouldn't derive the sensitivities from the exact same 
% dataset we apply them to, like we're doing here for demonstration purposes. 
% Now try performing the same set of reconstructions, but with the source image 
% shifted slightly in space to see the effect of over-fitting:

% R=4 SENSE reconstruction with S_0, S_1, and S_ref in the presence of
% slight image translation
img_R4_moved = circshift(circshift(img_R4,1,2),1,1);
img_S0b = SENSE(img_R4_moved,S_0,4);
img_S1b = SENSE(img_R4_moved,S_1,4);
img_S2b = SENSE(img_R4_moved,S_2,4);
img_SREFb = SENSE(img_R4_moved,S_ref,4);
show_grid(abs(cat(4,img_S0b,img_S1b,img_S2b,img_SREFb)),[0 16],gray)
%% 
% * What do you observe?
%% The Limits of Parallel Imaging
% Now that we have our SENSE implementation working, lets consider the limits 
% of parallel imaging. 
% 
% Try pushing the under-sampling factor to |R=6:|

% R=6 SENSE reconstruction
raw_R6 = raw;
idx_R6 = setdiff(1:Nx,1:6:Nx);
raw_R6(idx_R6,:,:,:) = 0;
img_R6 = ifft2c(raw_R6);
img_R6_SENSE = SENSE(img_R6,S_2,6);
show_img(abs(img_R6_SENSE),[0 16],gray);
%% 
% Or |R=8:|

% R=8 SENSE reconstruction
raw_R8 = raw;
idx_R8 = setdiff(1:Nx,1:8:Nx);
raw_R8(idx_R8,:,:,:) = 0;
img_R8 = ifft2c(raw_R8);
img_R8_SENSE = SENSE(img_R8,S_2,8);
show_img(abs(img_R8_SENSE),[0 16],gray);
%% 
% Along the same lines, what happens if you select a subset of the coil 
% information? Try a reconstruction at |R=4| using only |Nc=8| of the coil sensitivities 
% and corresponding images:

% Nc=8 SENSE reconstruction @ R=4
img_R4_Nc8 = SENSE(img_R4(:,:,:,1:8),S_2(:,:,:,1:8),4);
show_img(abs(img_R4_Nc8),[0 16],gray);
%% 
% Or |Nc=4:|

% Nc=4 SENSE reconstruction @ R=4
img_R4_Nc4 = SENSE(img_R4(:,:,:,1:4),S_2(:,:,:,1:4),4);
show_img(abs(img_R4_Nc4),[0 16],gray);
%% 
% * What happens to the reconstructions as |Nc| is close to or equal to |R? 
% |
% * How might you select an _optimal_ subset of |Nc=4| coils? What makes it 
% _optimal_?
% * What happens if you try to perform an |R=4| reconstruction with |Nc<R|?

% Optimal Nc=4 SENSE reconstruction @ R=4 using SVD-based coil compression
[~,~,v]=svd(reshape(S_2,[],16),0);
S_Nc4 = reshape(reshape(S_2,[],16)*v(:,1:4),[Nx,Ny,Nz,4]);
img_R4_Nc4 = reshape(reshape(img_R4,[],16)*v(:,1:4),[Nx,Ny,Nz,4]);
img_R4_Nc4opt = SENSE(img_R4_Nc4,S_Nc4,4);
show_img(abs(img_R4_Nc4opt),[0 16],gray);
%% 
% One aspect of parallel imaging that we have not discussed so far is noise, 
% and noise amplification in particular that results from SENSE (and other) reconstructions.
% 
% Because this entire reconstruction is linear we can analyse noise behaviour 
% separately. Try replacing the input data in your reconstructions with complex 
% Gaussian white noise, and observe the spatial distribution of noise in the output 
% images (use the reference sensitivities |S_ref| to get a cleaner picture). To 
% get a sense of the relative change in noise with |R|, we can show this noise 
% standard deviation relative to an |R=1| reconstruction:

% Noise SENSE reconstructions
outN4 = zeros(Nx,Ny,100);
outN1 = zeros(Nx,Ny,100);
for n = 1:100
    noise = (randn(Nx,Ny,Nz,Nc) + 1j*randn(Nx,Ny,Nz,Nc));
    outN4(:,:,n) = SENSE(noise,S_ref,4);
    outN1(:,:,n) = SENSE(noise,S_ref,1);
end
outN1(outN1==0)=inf; % prevents any division-by-zero
show_img(std(outN4,[],3)./(std(outN1,[],3)),[0 3],jet);
colorbar();
%% 
% * What do you observe happening to the noise of the output images as the under-sampling 
% factor increases? 
% 
% The spatial non-uniformity of the output noise is the so-called "geometry-factor", 
% which characterises noise-amplification due to the reconstruction. 
% 
% * Where is the noise amplification worst?
% * Can you relate this to some characteristic of the linear least squares sub-problems? 
% * How might you theoretically predict the amount of noise amplification you 
% expect to see?
% * How do you think this influences the choice of under-sampling factor?
%% Helper functions
%%
function out = SENSE(input,sens,R)
    [Nx,Ny,Nz,Nc] = size(input);
    out = zeros(Nx,Ny);
    % loop over the top-1/R of the image
    for x = 1:Nx/R
        x_idx = x:Nx/R:Nx;
        % loop over the entire left-right extent
        for y = 1:Ny
            % pick out the sub-problem sensitivities
            S = transpose(reshape(sens(x_idx,y,1,:),R,[]));
            % solve the sub-problem in the least-squares sense
            out(x_idx,y) = pinv(S)*reshape(input(x,y,1,:),[],1);
        end
    end
end


function show_grid(data, cscale, cmap)
    if nargin < 2
        cscale = [];
    end
    if nargin < 3
        cmap = gray;
    end
    figure();
    N = ndims(data);
    sz = size(data,N);
    n = ceil(sqrt(sz));
    m = ceil(sz/n);
    idx = repmat({':'},1,N);
    for i = 1:m
        for j = 1:n
            idx{N} = (i-1)*m+j;
            subplot('position',[(i-1)/m (n-j)/n (1/m-0.005) (1/n-0.005)]);
            imshow(data(idx{:}),cscale,'colormap',cmap);
        end
    end
end
%{
function show_img(data, cscale, cmap)
    if nargin < 2
        cscale = [];
    end
    if nargin < 3
        cmap = gray;
    end
    figure();
    imshow(data,cscale,'colormap',cmap);
end
%}
function show_img(data, cscale, cmap)
   if nargin < 2 || isempty(cscale)
       cscale = [-inf inf];
   end
   if nargin < 3
       cmap = gray;
   end
   figure();
   imagesc(data);
   axis equal
   colormap(cmap);
   caxis(cscale);
   plotH = gca;
   plotH.XTick = [];plotH.YTick = [];plotH.YColor = 'w';plotH.XColor = 'w';
end
function S = adaptive_est_sens(data)
    [Nx,Ny,Nz,Nc] = size(data);
    S = zeros(Nx,Ny,Nz,Nc);
    M = zeros(Nx,Ny,Nz);
    w = 5;
    for i = 1:Nx
        ii = max(i-w,1):min(i+w,Nx);
        for j = 1:Ny
            jj = max(j-w,1):min(j+w,Ny);
            for k = 1:Nz
                kk = max(k-w,1):min(k+w,Nz);
                kernel = reshape(data(ii,jj,kk,:),[],Nc);
                [V,D] = eigs(conj(kernel'*kernel),1);
                S(i,j,k,:) = V*exp(-1j*angle(V(1)));
                M(i,j,k) = sqrt(D);
            end
        end
    end
    S = S.*(M>0.1*max(abs(M(:))));
end
function out = fft2c(input)
    out = fftshift(fft(ifftshift(input,1),[],1),1);
    out = fftshift(fft(ifftshift(out,2),[],2),2);
end
function out = ifft2c(input)
    out = fftshift(ifft(ifftshift(input,1),[],1),1);
    out = fftshift(ifft(ifftshift(out,2),[],2),2);
end
