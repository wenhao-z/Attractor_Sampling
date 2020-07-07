% Test the performance of decentralized system under different parameters

% Wen-Hao Zhang, Oct-6-2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

setWorkPath;

% Load parameters
parsSingleCANN;

NetPars.AmplRatio = 0:0.1:2;
NetPars.JrcRatio = 0.5;
NetPars.fanoFactor = 0.5;

% Generate grid of parameters
[parGrid, dimPar] = paramGrid(NetPars);
parGrid = arrayfun(@(x) getDependentPars(x), parGrid);

%% Net Simulation
NetStat = struct('meanBumpPos', [], ...
    'mrlBumpPos', [], ...
    'concBumpPos', [], ...
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

%% Theoretical prediction

wfwd = 8/3^(3/2) * NetPars.fanoFactor * 2.5;

% Old code directly specifies the input strength
% PreMat_LH = sqrt(2*pi) * [parGrid.Ampl] .* NetPars.rho./ (sqrt(2)*NetPars.TunWidth);

% New version includes the spatial convolution of the feedforward input.
PreMat_LH = 2*sqrt(pi) * [parGrid.Ampl] / NetPars.TunWidth/ wfwd;

% ------------------------------------------------------
% The variance of noise is the firing rate bump 
% varBumpPosTheory = [NetStat.OHeightAvg]./ [NetStat.UHeightAvg]*NetPars.TunWidth ...
%     ./ (2*sqrt(pi) .*[parGrid.Ampl]);

% The variance of noise is the synaptic input bump
varBumpPosTheory = 4*NetPars.fanoFactor*NetPars.TunWidth/sqrt(pi)/ 3^(3/2) ./ [parGrid.Ampl];

%%
cMap = lines(4);

figure
subplot(1,2,1)
plot(NetPars.AmplRatio, 1./reshape([NetStat.varBumpPos], size(parGrid)), 'o', 'color', cMap(1,:))
hold on
plot(NetPars.AmplRatio, 1./reshape(varBumpPosTheory, size(parGrid)), 'color', cMap(1,:))
plot(NetPars.AmplRatio, reshape(PreMat_LH, size(parGrid)), 'color', cMap(2,:))
xlabel('Input intensity')
title(['JrcRatio=', num2str(NetPars.JrcRatio)])
ylabel('Precision of samples')
legend('1./Bump posi. var. (sim)', '1./Bump posi. var. (theory)', 'Fisher info.', 'location', 'best')
axis square

subplot(1,2,2)
% plot(NetPars.AmplRatio, [NetStat.OHeightAvg])
hold on
plot(NetPars.AmplRatio, reshape([NetStat.UHeightAvg], size(parGrid))./ ...
    reshape([NetStat.OHeightAvg], size(parGrid)),'o', 'color', cMap(1,:))
plot(NetPars.AmplRatio, ...
    1./ reshape([NetStat.UHeightAvg], size(parGrid)) + ...
    sqrt(2*pi)*NetPars.k*NetPars.rho*NetPars.TunWidth* ...
    reshape([NetStat.UHeightAvg], size(parGrid)), ...
    'color', cMap(1,:))
ylabel('U/R')

yyaxis right
plot(NetPars.AmplRatio, [NetStat.UHeightAvg], 'color', cMap(2,:))
hold on
plot(NetPars.AmplRatio, [NetStat.OHeightAvg], 'color', cMap(3,:))
ylabel('Bump height')

legend('Sim.', 'Theory', 'U height (sim)', 'R height (sim)', 'location', 'best')
axis square
% legend('1./Bump posi. var.', 'Fisher info.', 'Rate average','location', 'best')

%%
figure
subplot(1,2,1)
plot(NetPars.PrefStim, mean(InputSet.O(:,:,NetPars.tStat+1:end),3))
hold on
% plot(NetPars.PrefStim, std(InputSet.O(:,:,NetPars.tStat+1:end), 0, 3))
plot(NetPars.PrefStim, var(InputSet.O(:,:,NetPars.tStat+1:end), 0, 3))
axis square 

subplot(1,2,2)
plot(NetPars.PrefStim, var(InputSet.O(:,:,NetPars.tStat+1:end), 0, 3)./ mean(InputSet.O(:,:,NetPars.tStat+1:end),3))
ylabel('Fano factor')
axis square