% Test the performance of decentralized system under different parameters

% Wen-Hao Zhang, Oct-6-2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

setWorkPath;

% Load parameters
parsSingleCANN;

NetPars.N = 180;
NetPars.AmplRatio = 1;
NetPars.JrcRatio = 1;

NetPars.tTrial  = 100 * NetPars.tau;
NetPars.nTrials = 1; % number of trials
NetPars.tLen    = NetPars.nTrials * NetPars.tTrial;

NetPars.seedNois = sum(clock);

NetPars.JRand = 5;

parseNetPars;
% Generate grid of parameters
[parGrid, dimPar] = paramGrid(NetPars);
parGrid = arrayfun(@(x) getDependentPars(x), parGrid);

%% Net Simulation
NetStat = struct('BumpPos', [], ...
    'meanBumpPos', [], ...
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
    [InputSet, NetStat(iterPar)] = simChaoticAttractors(InputSet, netpars, outArgs);
end

tEnd = clock;

%% Theoretical prediction

wfwd = 8/3^(3/2) * NetPars.fanoFactor * 2.5;

% Old code directly specifies the input strength
% PreMat_LH = sqrt(2*pi) * [parGrid.Ampl] .* NetPars.rho./ (sqrt(2)*NetPars.TunWidth);

% New version includes the spatial convolution of the feedforward input.
PreMat_LH = 2*sqrt(pi) * [parGrid.Ampl] / NetPars.TunWidth/ wfwd;

% ------------------------------------------------------------
% The variance of noise is the firing rate bump 
% varBumpPosTheory = [NetStat.OHeightAvg]./ [NetStat.UHeightAvg]*NetPars.TunWidth ...
%     ./ (2*sqrt(pi) .*[parGrid.Ampl]);

% The variance of noise is the synaptic input bump
varBumpPosTheory = 4*NetPars.fanoFactor*NetPars.TunWidth/sqrt(pi)/ 3^(3/2) ./ [parGrid.Ampl];

% ------------------------------------------------------------
% Auto correlation function.

[CCFunc, tLag] = xcorr(NetStat.BumpPos(1,NetPars.tStat/NetPars.dt+1:end), 1e3);
CCFunc = CCFunc((end+1)/2:end)./max(CCFunc);
tLag = tLag((end+1)/2:end) * NetPars.dt;
CCFunc_Theory = exp(-abs(tLag)*parGrid.Ampl/NetPars.tau/NetStat.UHeightAvg);

%%
figure 
hAxe(1) = subplot(3,4,1:3);
tPlot = 5e3;
plot((1:tPlot)*NetPars.dt, NetStat.BumpPos(NetPars.tStat/NetPars.dt+(1:tPlot)) );
xlabel('Time (\tau)')
ylabel('Sample s')

hAxe(2) = subplot(3,4,4);
[ProbSample, edgeSample] = histcounts(NetStat.BumpPos(NetPars.tStat/NetPars.dt+1:end), 1e2);
stairs(ProbSample/sum(ProbSample), (edgeSample(1:end-1)+edgeSample(2:end))/2)
hold on
plot(normpdf((edgeSample(1:end-1)+edgeSample(2:end))/2, NetPars.Posi, 1/sqrt(PreMat_LH))*mean(diff(edgeSample)), ...
    (edgeSample(1:end-1)+edgeSample(2:end))/2)
% plot(normpdf(edgeSample, NetPars.Posi, sqrt(NetStat.varBumpPos))*mean(diff(edgeSample)), edgeSample)
xlabel('Prob. of samples')

linkaxes(hAxe, 'y')

subplot(3,1,2:3)
plot(tLag, CCFunc)
hold on
plot(tLag, CCFunc_Theory)
axis square
ylim([-0.01, 1])
legend('Sim', 'Theory')
xlabel('Time (\tau)')
ylabel('Cross correlation')
set(gca, 'ytick', 0:0.25:1)
title(['AmplRatio=', num2str(NetPars.AmplRatio), ', JrcRatio=', num2str(NetPars.JrcRatio)])

%%
figure

subplot(2,2,1:2)
tPlot = NetPars.tLen/NetPars.dt;
imagesc((1:tPlot)*NetPars.dt, NetPars.PrefStim, squeeze(InputSet.O(:,1,1:tPlot)) );
axis xy
xlabel('Time (\tau)')
ylabel('Neuron index \theta')
set(gca, 'ytick', [-178, 0, 180], 'yticklabel', [-180, 0, 180])

subplot(2,3,4)
hold on
plot(NetPars.PrefStim, NetStat.OAvgXTime)
% plot(NetPars.PrefStim, NetStat.OAvgXTime - NetStat.OStdXTime)
% plot(NetPars.PrefStim, NetStat.OAvgXTime + NetStat.OStdXTime)
set(gca, 'xtick', NetPars.Width * (-1:0.5:1))
xlim(NetPars.Width * [-1, 1])
ylim([0, 50])
ylabel('Firing rate(Hz)')
xlabel('Neuron index \theta')
axis square 
title(['AmplRatio=', num2str(NetPars.AmplRatio), ', JrcRatio=', num2str(NetPars.JrcRatio)])

subplot(2,3,5)
plot(NetPars.PrefStim, NetStat.OStdXTime.^2 ./ NetStat.OAvgXTime);
set(gca, 'xtick', NetPars.Width * (-1:0.5:1))
xlim(NetPars.Width * [-1, 1])
ylabel('Fano factor')
xlabel('Neuron index \theta')
box off
axis square

subplot(2,3,6)
hist(NetStat.OStdXTime.^2 ./ NetStat.OAvgXTime, 50);
xlabel('Fano factor')
ylabel('Histogram')
box off
axis square
