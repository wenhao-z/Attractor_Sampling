function [Lambda_s, KLD] = findPriorPrecisionHD(muS, covS, muLH, invCovLH)
% Find the prior precision stored in the neural network given
% 1) the sample distribution generated by network dynamics, and the
% 2) likelihood parameters

% Prior precision: Lambda_s

% Input:
% 1) muLH: the mean of likelihood, calculated from theory,
% 2) invCovLH: the precision matrix of likelihood.

% Output:
% 1) Lambda_s: the precision of the Laplacian precision matrix in
%              prior, which denotes the correlation between two stimuli;
% 2) KLD: the KL divergence from the searched posterior and the actual
%         sample distribution.

% Wen-Hao Zhang, Apr. 15, 2020
% University of Pittsburgh


% The initial value in optimization
invCovPost0 = inv(covS);
maxLambda_s = 2*invCovPost0(1);
% Lambda_s0 = - invCovPost0(1,2);
% Lambda_s0 = -(invCovPost0 - diag(invCovPost0));

Lambda_s0 = invCovPost0 - diag(invCovPost0) - diag(sum(invCovPost0,2));

% Find the Lambda_s making the posterior closet to the sample distribution
options = optimset('TolX', 1e-4, 'display', 'off');
Lambda_s = fmincon(@(lambda) computeKLD(lambda), Lambda_s0, [], [], [], [], ...
    0, maxLambda_s, [], options);
KLD = computeKLD(Lambda_s);


    function KLDiv = computeKLD(Lambda_s)
        % The precision matrix of the posterior
        % invCovPost = 2*diag(ones(1, length(invCovLH))) - ones(size(invCovLH));
        % invCovPost = Lambda_s * invCovPost;
        
%         Lambda_s = diag(sum(Lambda_s,2)) + Lambda_s;
        invCovPost = Lambda_s + invCovLH;
        
        % The mean of the posterior
        muPost = invCovPost \ invCovLH * muLH;
        
        KLDiv = getKLDiv(muPost, inv(invCovPost), muS, covS);
    end
end