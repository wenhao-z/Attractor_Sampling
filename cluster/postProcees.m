%% Parameters of sample distributions
PreMat_Sample = reshape({NetStat.varBumpPos}, size(JMatArray));
PreMat_Sample = cellfun(@inv, PreMat_Sample, 'uniformout', 0);
% PreMat_Sample = reshape(cell2mat(PreMat_Sample), NetPars.numNets, NetPars.numNets, []);

%% Theoretical prediction
EmpCorrectFactor = 2.5; % Empirical correction factor to predict the position fluctuation

wfwd = 8/3^(3/2) * NetPars.fanoFactor * EmpCorrectFactor;

% New version includes the spatial convolution of the feedforward input.
% PreMat_LH = cell(size(PreMat_Sample));
PreMat_LH = cellfun(@(x) 2*sqrt(pi) * diag(x) *NetPars.Uc/ NetPars.TunWidth/ wfwd, ...
    InputAmpl, 'uniformout', 0);

% The variance of noise is the synaptic input bump
% varBumpPosTheory = 4*NetPars.fanoFactor*NetPars.TunWidth/sqrt(pi)/ 3^(3/2) ./ [NetPars.Ampl];

% ------------------------------------------------------
% Prediction of posterior distributions
meanSampleTheory = cell(size(JMatArray));
PreMatSampleTheory = cell(size(JMatArray));
AntiSymConns = cell(size(NetStat));
AntiSymConns1 = cell(size(NetStat));
eigVal_SampleDyn = cell(size(NetStat));

for iterPar = 1: numel(NetStat)
    % Theoretical prediction of posterior mean
    meanSampleTheory{iterPar} = NetStat(iterPar).varBumpPos * ...
        PreMat_LH{iterPar} * InputPosi{iterPar}; %NetPars.Posi;
    
    % Functional connection strength (Note the TRANSPOSE of JMatArray!!!)
    FuncConns = JMatArray{iterPar}' .* NetStat(iterPar).OHeightAvg';
    FuncConns = FuncConns.*NetPars.rho*sqrt(2*pi)/NetPars.TunWidth/wfwd;
    
    % Theoretical prediction of posterior precision matrix
    PreMat_Post = -(FuncConns + FuncConns')/2;
    PreMat_Post = PreMat_Post - diag(diag(PreMat_Post));
    PreMat_Post = PreMat_Post - diag(sum(PreMat_Post));
    PreMat_Post = PreMat_Post + PreMat_LH{iterPar};
    PreMatSampleTheory{iterPar} = PreMat_Post;
    
    % Antisymmetric connections in the network
    AntiSymConns1{iterPar} = (FuncConns - FuncConns')/2;
    AntiSymConns{iterPar} = (FuncConns + PreMat_LH{iterPar}) * NetStat(iterPar).varBumpPos ...
        + eye(size(FuncConns));
    
    % Effective interaction matrix between stimulus feature samples
    IntracMat = FuncConns;
    IntracMat = IntracMat - diag(diag(IntracMat));
    IntracMat  = IntracMat - diag(sum(IntracMat,2));
    IntracMat = IntracMat ./ ...
        kron(NetStat(iterPar).UHeightAvg(:), ones(1, length(FuncConns)) );
    IntracMat = IntracMat * NetPars.rho / sqrt(2);
    eigVal_SampleDyn{iterPar} = eig(IntracMat);
end
clear PreMat_Post PreMat_Post FuncConns

%%
cMap = lines(4);
figure

subplot(2,3,1)
hold on
for iterNet = 1: length(numNetsArray)
    plot(reshape([NetStat(:,iterNet).meanBumpPos],[],1), ...
        reshape([meanSampleTheory{:,iterNet}], [],1), '.', 'markersize', 4)
end
% xyLim = max(abs(InputPosi(:)));
xyLim = 10;
plot(xyLim*[-1,1], xyLim*[-1,1], '--k')
xlabel('Mean of samples')
ylabel('Mean (theory)')
axis square
title(['EmpCorrectFactor=' num2str(EmpCorrectFactor)])

subplot(2,3,2)
hold on

for iterPar = 1: nMCSim*length(numNetsArray)
    Omega = PreMat_Sample{iterPar}; % Posterior precision matrix
    xDat = -reshape(triu(Omega,1), 1,[]);
    
    % Theoretical prediction of diagonal elements of prior precision
    yDat = -PreMatSampleTheory{iterPar};
    
    plot(xDat(xDat~=0), yDat(xDat~=0), '.', 'markersize', 4)
end
clear xDat yDat

xyLim = axis(gca);
xyLim = [min(xyLim([1,3])), max(xyLim([2,4]))];
plot(xyLim, xyLim, '--k')
axis([xyLim, xyLim])

xlabel('Prior precision (searched)')
ylabel('Prior precision (theory)')
axis square
clear xyLim

% -------------------------------------------------------------------------
% Compare the precision of stimulus samples
subplot(2,3,3)
hold on
for iterPar = 1: nMCSim*length(numNetsArray)
    Omega = PreMat_Sample{iterPar}; % Posterior precision matrix
    xDat = diag(Omega);
    yDat = diag(PreMatSampleTheory{iterPar});
    
    plot(xDat, yDat, '.', 'markersize', 4)
end
xyLim = axis(gca);
xyLim = [min(xyLim([1,3])), max(xyLim([2,4]))];
plot(xyLim, xyLim, '--k')
axis([xyLim, xyLim])
xlabel('Precision of samples')
ylabel('Posterior precision')
axis square

% -------------------------------------------------------------------------
% Plot the antisymmetric couplings' influence on sampling distribution

subplot(2,3,4)
hold on

for iterPar = 1: nMCSim*length(numNetsArray)
    dSampleMean = NetStat(iterPar).meanBumpPos - meanSampleTheory{iterPar};
    dSampleMean_Pred = PreMatSampleTheory{iterPar}\ ...
        AntiSymConns1{iterPar} * meanSampleTheory{iterPar};
    
    plot(dSampleMean, dSampleMean_Pred, '.', 'markersize', 4, 'color', cMap(1,:))
end
plot(4*[-1, 1], 4*[-1, 1], '--k')
axis square
xlim([-4,4])
xlabel('Deviation of mean (actual')
ylabel('Deviation of mean (prediction)')


subplot(2,3,5)
hold on
for iterPar = 1: nMCSim*length(numNetsArray)
    dPreMat = PreMatSampleTheory{iterPar} - PreMat_Sample{iterPar};
        
    Commutator = AntiSymConns1{iterPar} * PreMatSampleTheory{iterPar} ...
        - PreMatSampleTheory{iterPar} * AntiSymConns1{iterPar};
    
    plot(diag(Commutator), diag(dPreMat), '.', 'markersize', 4, 'color', cMap(1,:))
%     plot(diag(dPreMat), diag(dPreMat_Pred),'.', 'markersize', 4, 'color', cMap(1,:))
    
    xDat = triu(dPreMat,1);
    yDat = triu(Commutator,1);
%     yDat = triu(dPreMat_Pred,1);
    plot(xDat(xDat~=0), yDat(xDat~=0), '.', 'markersize', 4, 'color', cMap(2,:))
end

plot([-1,1], zeros(1,2), '--k')
plot(zeros(1,2), [-1,1], '--k')

xlabel('Commutator')
ylabel('Deviation of Precision')
axis square

axis([-1 1 -1 1])
clear xDat yDat


% -------------------------------------------------------------------------
% Plot the slowest eigenvalue with the antisymmetric coupling
subplot(2,3,6)
hold on
xDat = [];
yDat = [];
for iterPar = 1: nMCSim*length(numNetsArray)
    xDat = [xDat, sqrt(sum(AntiSymConns1{iterPar}(:).^2))];
    yDat = [yDat, 1./-min((real(eigVal_SampleDyn{iterPar})))];
end

ft = fittype('a*x^b');
fitObj = fit(xDat', log(yDat'), ft);

plot(xDat, yDat, '.', 'markersize', 4, 'color', cMap(1,:));
% plot(fitObj, xDat, yDat)
plot(sort(xDat), exp(feval(fitObj, sort(xDat))))
ylabel('Slowest sampling time constant')
xlabel('Norm of the antisymmetric coupling')
set(gca, 'yscale', 'log', 'ylim', [30, 1e3])
axis square


%% Deviation of sampling precision (new method)
figure
hold on
for iterPar = 1: nMCSim*length(numNetsArray)
    dCovMat = NetStat(iterPar).varBumpPos - inv(PreMatSampleTheory{iterPar});
    dCovMat = PreMatSampleTheory{iterPar} * dCovMat + ...
        dCovMat * PreMatSampleTheory{iterPar};
        
    dPreMat_Pred = NetStat(iterPar).varBumpPos * AntiSymConns1{iterPar} ...
        - AntiSymConns1{iterPar} * NetStat(iterPar).varBumpPos;
    
    plot(diag(dCovMat), diag(dPreMat_Pred),'.', 'markersize', 4, 'color', cMap(1,:))
    
    xDat = triu(dCovMat,1);
    yDat = triu(dPreMat_Pred,1);
    plot(xDat(xDat~=0), yDat(xDat~=0), '.', 'markersize', 4, 'color', cMap(2,:))
end

axis([-1 1 -1 1])
plot([-1 1], [-1,1], '--k')