% Test the performance of decentralized system under different parameters

% Wen-Hao Zhang, Oct-6-2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

setWorkPath;

% Load parameters
parsCoupledCANNs;

NetPars.AmplRatio = 1*ones(NetPars.numNets,1);
NetPars.JrcRatio = 0.5;
NetPars.JrpRatio = 0:0.1:2;

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

wfwd = 8/3^(3/2) * NetPars.fanoFactor * 2.5;

% New version includes the spatial convolution of the feedforward input.
PreMat_LH = 2*sqrt(pi) * diag(parGrid(1).Ampl) / NetPars.TunWidth/ wfwd;

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
       PreMat_LH * NetPars.Posi;
   [Lambda_s(iter), KLD(iter)] = findPriorPrecision(NetStat(iter).meanBumpPos, NetStat(iter).varBumpPos, ...
    NetPars.Posi, PreMat_LH);
end
meanSampleTheory = reshape(meanSampleTheory, [NetPars.numNets, size(parGrid)]);

%%
cMap = lines(4);

figure

subplot(2,2,1)
hold on
plot([parGrid.Jrp], [NetStat.meanBumpPos], 'o')
plot([parGrid.Jrp], meanSampleTheory)
xlabel('Jrp')
ylabel('Mean of samples')
axis square 
title(['JrcRatio=', num2str(NetPars.JrcRatio)]);

subplot(2,2,2)
hold on
plot([parGrid.Jrp], Lambda_s)
plot([parGrid.Jrp], [parGrid.Jrp].*mean([NetStat.OHeightAvg],1).*NetPars.rho*sqrt(2*pi) ...
    /NetPars.TunWidth/wfwd, 'o')

% plot([parGrid.Jrp], Lambda_s .* NetPars.TunWidth/ NetPars.rho/sqrt(2*pi) ...
%     * wfwd ./ mean([NetStat.OHeightAvg],1), '-o');
% plot([0, max([parGrid.Jrp])], [0, max([parGrid.Jrp])], '--k')
xlabel('Jrp')
ylabel('Prior precision (searched)')

yyaxis right
plot([parGrid.Jrp], KLD)
ylabel('KL Divergence')
axis square 

subplot(2,2,3)
hold on
plot(NetPars.JrpRatio, squeeze(PreMat_Sample(1,1,:)) - squeeze(PreMat_Sample(1,1,1)) )
plot(NetPars.JrpRatio, squeeze(PreMat_Sample(2,2,:)) - squeeze(PreMat_Sample(2,2,1)))
plot(NetPars.JrpRatio, -squeeze(PreMat_Sample(1,2,:)))
xlabel('Jrp')
ylabel('Prior precision')
legend('Prediction (Net 1)', 'Prediction(From Net 2)', 'Corr.(Net1, Net2)')
axis square 
