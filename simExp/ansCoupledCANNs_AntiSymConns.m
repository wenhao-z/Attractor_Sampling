% Test the performance of decentralized system under different parameters

% Wen-Hao Zhang, Feb 22, 2021
% wenhaoz@uchicago.edu
% University of Chicago

setWorkPath;

% Load parameters
parsCoupledCANNs;

NetPars.AmplRatio = 1*ones(NetPars.numNets,1);
NetPars.JrcRatio = 0.5;
NetPars.meanJrpRatio = 0.8; 
NetPars.diffJrpRatio = 0:0.1:1;

NetPars.fanoFactor = 0.5;
NetPars.Posi = 2*[-1;1];

% Generate grid of parameters
[parGrid, dimPar] = paramGrid(NetPars);
parGrid = arrayfun(@(x) getDependentPars(x), parGrid);

%% Net Simulation
NetStat = struct('meanBumpPos', [], ...
    'varBumpPos', [], ...
    'OHeightAvg', [], ...
    'OAvgXTime', [], ...
    'OStdXTime', [], ...
    'UHeightAvg', []);
NetStat = repmat(NetStat, size(parGrid));

tStart = clock;
for iterPar = 1: numel(parGrid)
    fprintf('Progress: %d/%d\n', iterPar, numel(parGrid));
    netpars = parGrid(iterPar);
    
    % Generating the connection matrix with certain anti-symmetric coupling
    JMat = eye(NetPars.numNets) + ...
        rot90(diag(1+[-1,1]*netpars.diffJrpRatio/2)) * netpars.meanJrpRatio;
    JMat = JMat * netpars.Jrc;
    netpars.JMat = JMat;
    
    % Network input
    InputSet = makeNetInput([], netpars);
    
    % Run simulation
    outArgs = struct('InputSet', [], 'NetStat', NetStat(iterPar));
    [InputSet, NetStat(iterPar)] = simCoupledAttractors1(InputSet, netpars, outArgs);
end

tEnd = clock;

%% Parameters of sample distributions
PreMat_Sample = {NetStat.varBumpPos};
PreMat_Sample = cellfun(@inv, PreMat_Sample, 'uniformout', 0);
PreMat_Sample = reshape(cell2mat(PreMat_Sample), NetPars.numNets, NetPars.numNets, []);

%% Theoretical prediction

EmpCorrectFactor = 2.8; % Empirical correction factor to predict the position fluctuation

wfwd = 8/3^(3/2) * NetPars.fanoFactor * EmpCorrectFactor;

% New version includes the spatial convolution of the feedforward input.
PreMat_LH = zeros(NetPars.numNets, NetPars.numNets, numel(parGrid));
for iter = 1: numel(parGrid)
    PreMat_LH(:,:,iter) = 2*sqrt(pi) * diag(parGrid(iter).Ampl) / NetPars.TunWidth/ wfwd;
end

% ------------------------------------------------------
% The variance of noise is the firing rate bump
% varBumpPosTheory = [NetStat.OHeightAvg]./ [NetStat.UHeightAvg]*NetPars.TunWidth ...
%     ./ (2*sqrt(pi) .*[parGrid.Ampl]);

% The variance of noise is the synaptic input bump
varBumpPosTheory = 4*NetPars.fanoFactor*NetPars.TunWidth/sqrt(pi)/ 3^(3/2) ./ [parGrid.Ampl];

% ------------------------------------------------------
% Prediction of mean of samples
% Find the prior precision which best explains the sample distribution
meanSampleTheory = zeros(NetPars.numNets, numel(parGrid));
Lambda_s = zeros(1, numel(parGrid));
KLD = zeros(1, numel(parGrid));

for iter = 1: numel(NetStat)
    meanSampleTheory(:,iter) = NetStat(iter).varBumpPos * ...
        PreMat_LH(:,:,iter) * NetPars.Posi;
    [Lambda_s(iter), KLD(iter)] = findPriorPrecision(NetStat(iter).meanBumpPos, ...
        NetStat(iter).varBumpPos, NetPars.Posi, PreMat_LH(:,:,iter));
end
meanSampleTheory = reshape(meanSampleTheory, [NetPars.numNets, size(parGrid)]);

% Find the prior precision in another way (only work for TWO coupled nets)
Lambda_s = zeros(1, numel(parGrid));
for iterPar = 1: numel(NetStat)
    Omega = inv(NetStat(iterPar).varBumpPos); % Posterior precision matrix
    Lambda_s(iterPar) = -Omega(1,2);
end
clear Omega iterPar

%%
cMap = lines(4);
figure

subplot(1,2,1)
hold on
plot(NetPars.diffJrpRatio, [NetStat.meanBumpPos], 'o')
plot(NetPars.diffJrpRatio, meanSampleTheory)
xlabel('% diff. of Jrp')
ylabel('Mean of samples')
axis square
title({['JrcRatio=', num2str(NetPars.JrcRatio)], ...
    ['JrpRatioAvg=', num2str(NetPars.meanJrpRatio)]});


subplot(1,2,2)
hold on
% Predictition of the Prior precision
Lambda_sNetPred = NetPars.Jrc*NetPars.meanJrpRatio * ...
    [1 - NetPars.diffJrpRatio/2; 1 + NetPars.diffJrpRatio/2];
Lambda_sNetPred = mean(Lambda_sNetPred .* [NetStat.OHeightAvg], 1);
Lambda_sNetPred = Lambda_sNetPred .*NetPars.rho*sqrt(2*pi)/NetPars.TunWidth/wfwd;

plot(NetPars.diffJrpRatio, squeeze(PreMat_Sample(1,1,:)), 'o', 'color', cMap(1,:))
plot(NetPars.diffJrpRatio, squeeze(PreMat_Sample(2,2,:)), 'o', 'color', cMap(2,:))
plot(NetPars.diffJrpRatio, Lambda_s, 'o', 'color', cMap(3,:))

plot(NetPars.diffJrpRatio, squeeze(PreMat_LH(1,1,:))'+Lambda_s, 'color', cMap(1,:))
plot(NetPars.diffJrpRatio, squeeze(PreMat_LH(2,2,:))'+Lambda_s, 'color', cMap(2,:))
plot(NetPars.diffJrpRatio, Lambda_sNetPred, 'color', cMap(3,:))
        
xlabel('% diff. of Jrp')
ylabel('Prior precision (searched)')
axis square
