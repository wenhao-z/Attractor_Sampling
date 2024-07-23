% Test the performance of decentralized system under different parameters

% Wen-Hao Zhang, Oct-6-2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

setWorkPath;

% Load parameters
parsSingleCANN;

flagTest = 2;
% 1: Change the input intensity while fix the recurrent weight
% 2: Change the recurrent weight while fix the input intensity
% switch flagTest
%     case 1
%         NetPars.AmplRatio = 0:0.1:2;
%         NetPars.JrcRatio = 0.5;
%     case 2
%         NetPars.AmplRatio = [0.4, 0.8, 1.2];
%         NetPars.JrcRatio = 0:0.1:0.9;
% end

NetPars.k = (0.1:0.2:1.5) * 5e-4;
NetPars.fanoFactor = 0.5;
NetPars.tTrial = 5e3 * NetPars.tau;

% Generate grid of parameters
[parGrid, dimPar] = paramGrid(NetPars);
% parGrid = arrayfun(@(x) getDependentPars(x), parGrid);

%% Net Simulation
NetStat = struct('BumpPos', [], ...
    'meanBumpPos', [], ...
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
EmpCorrectFactor = 2.8; % Empirical correction factor to predict the position fluctuation

wfwd = 8/3^(3/2) * NetPars.fanoFactor * EmpCorrectFactor;

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

% ------------------------------------------------------
% Decaying time constant of cross correlation function
tau_CC = zeros(size(parGrid));
tau_CC_Theory = zeros(size(parGrid));
for iter = 1: numel(NetStat)
    [CCFunc, tLag] = xcorr(NetStat(iter).BumpPos(NetPars.tStat/NetPars.dt+1:end), 1e3);
    CCFunc = CCFunc((end+1)/2:end)./max(CCFunc);
    tLag = tLag((end+1)/2:end) * NetPars.dt;
    
    NetStat(iter).CCFunc = CCFunc;
    fitObj = fit(tLag(1:4e2)', CCFunc(1:4e2)', 'exp1');
    tau_CC(iter) = - 1./fitObj.b;
    
    tau_CC_Theory(iter) = sqrt(2) * NetStat(iter).UHeightAvg ...
        / NetPars.rho/ parGrid(iter).Ampl / EmpCorrectFactor;
end

%%
cMap = lines(4);

figure
switch flagTest
    case 1
        subplot(1,3,1)
        plot(NetPars.AmplRatio, 1./reshape([NetStat.varBumpPos], size(parGrid)), 'o', 'color', cMap(1,:))
        hold on
        plot(NetPars.AmplRatio, 1./reshape(varBumpPosTheory, size(parGrid)), 'color', cMap(1,:))
        plot(NetPars.AmplRatio, reshape(PreMat_LH, size(parGrid)), 'color', cMap(2,:))
        xlabel('Input intensity')
        title([{['JrcRatio=', num2str(NetPars.JrcRatio)]} , ...
            {['EmpCorrCoef=' num2str(EmpCorrectFactor)] }])
        ylabel('Precision of samples')
        legend('1./Bump posi. var. (sim)', '1./Bump posi. var. (theory)', 'Fisher info.', 'location', 'best')
        axis square
        
        subplot(1,3,2)
        
        plot([NetStat.UHeightAvg]./NetPars.AmplRatio, tau_CC, 'o', 'color', cMap(1,:))
        hold on
        plot([NetStat.UHeightAvg]./NetPars.AmplRatio, tau_CC_Theory,'color', cMap(1,:))
        xlabel('Network gain, U/\alpha')
        ylabel('Time constant of xcorr. (/\tau)')
        
        axis square
        subplot(1,3,3)
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
    case 2
        subplot(1,3,1)
        plot(NetPars.JrcRatio, 1./reshape([NetStat.varBumpPos], size(parGrid)), 'o', 'color', cMap(1,:))
        hold on
        %plot(NetPars.JrcRatio, 1./reshape(varBumpPosTheory, size(parGrid)), 'color', cMap(1,:))
        plot(NetPars.JrcRatio, reshape(PreMat_LH, size(parGrid)), 'color', cMap(2,:))
        xlabel('Recurrent weight')
        title([{['AmplRatio=', num2str(NetPars.AmplRatio)]} , ...
            {['EmpCorrCoef=' num2str(EmpCorrectFactor)] }])
        ylabel('Precision of samples')
        legend('1./Bump posi. var. (sim)', '1./Bump posi. var. (theory)', 'Fisher info.', 'location', 'best')
        axis square
        
        subplot(1,3,2)
        plot(reshape([NetStat.UHeightAvg], size(NetStat))./NetPars.AmplRatio', tau_CC, 'o', 'color', cMap(1,:))
        hold on
        plot(reshape([NetStat.UHeightAvg], size(NetStat))./NetPars.AmplRatio', tau_CC_Theory,'color', cMap(1,:))
        xlabel('Network gain, U/\alpha')
        ylabel('Time constant of xcorr. (/\tau)')
        axis square
        
        subplot(1,3,3)
        plot(NetPars.JrcRatio, reshape([NetStat.UHeightAvg], size(NetStat)))
        axis square
        xlabel('Recurrent weight')
        ylabel('U bump height')
end
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