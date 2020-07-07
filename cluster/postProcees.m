%% Parameters of sample distributions
PreMat_Sample = reshape({NetStat.varBumpPos}, size(JMatArray));
PreMat_Sample = cellfun(@inv, PreMat_Sample, 'uniformout', 0);
% PreMat_Sample = reshape(cell2mat(PreMat_Sample), NetPars.numNets, NetPars.numNets, []);

%% Theoretical prediction

wfwd = 8/3^(3/2) * NetPars.fanoFactor * 2.5;

% New version includes the spatial convolution of the feedforward input.
% PreMat_LH = cell(size(PreMat_Sample));
PreMat_LH = cellfun(@(x) 2*sqrt(pi) * diag(x) *NetPars.Uc/ NetPars.TunWidth/ wfwd, ...
    InputAmpl, 'uniformout', 0);

% The variance of noise is the synaptic input bump
% varBumpPosTheory = 4*NetPars.fanoFactor*NetPars.TunWidth/sqrt(pi)/ 3^(3/2) ./ [NetPars.Ampl];

% ------------------------------------------------------
% Prediction of mean of samples
meanSampleTheory = cell(size(JMatArray));
for iterPar = 1: numel(NetStat)
    meanSampleTheory{iterPar} = NetStat(iterPar).varBumpPos * ...
        PreMat_LH{iterPar} * InputPosi{iterPar}; %NetPars.Posi;
end

%%
figure

subplot(1,2,1)
hold on
for iterNet = 1: length(numNetsArray)
    plot(reshape([NetStat(:,iterNet).meanBumpPos],[],1), ...
    reshape([meanSampleTheory{:,iterNet}], [],1), 'o', 'markersize', 4)
end
% xyLim = max(abs(InputPosi(:)));
xyLim = 10;
plot(xyLim*[-1,1], xyLim*[-1,1], '--k')
xlabel('Mean of samples')
ylabel('Mean (theory)')
axis square

subplot(1,2,2)
hold on


for iterPar = 1: nMCSim*length(numNetsArray)
    Omega = inv(NetStat(iterPar).varBumpPos); % Posterior precision matrix
    
    xDat = -reshape(triu(Omega,1), 1,[]);
    %     yDat = reshape(triu(JMatArray(:,:,iterPar),1).* NetStat(iterPar).OHeightAvg', 1,[])...
    %         .*NetPars.rho*sqrt(2*pi)/NetPars.TunWidth/wfwd;
    
    % Considering the
    yDat = JMatArray{iterPar}.* NetStat(iterPar).OHeightAvg';
    yDat = (yDat + yDat')/2;
    yDat = reshape(triu(yDat,1), 1,[]).*NetPars.rho*sqrt(2*pi)/NetPars.TunWidth/wfwd;
    
    plot(xDat(xDat~=0), yDat(xDat~=0), 'o', 'markersize', 4)
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