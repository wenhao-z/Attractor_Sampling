% Re-calculate some model parameters according to updated parameters
% Wen-Hao Zhang, Oct-06, 2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

%% Check the size of Input variables
if (size(NetPars.AmplRatio, 1) ~= NetPars.numNets)
    error('The dim. of NetPars.Ampl does not match with num. of nets.')
end

if (size(NetPars.Posi, 1) ~= NetPars.numNets)
    error('The dim. of NetPars.Posi does not match with num. of nets.')
end

% if strcmp(NetPars.typeExtNois, 'Gaussian')
%     if (size(NetPars.stdExtNois, 1) ~= NetPars.numNets)
%         error('The dim. of stdExtNois does not match with num. of nets.')
%     end
% end

if length(NetPars.PrefStim) ~= NetPars.N   
    PrefStim         = linspace(-NetPars.Width,NetPars.Width, NetPars.N+1)';
    PrefStim(1)      = [];
    NetPars.PrefStim = PrefStim;
    clear PrefStim
end

if (size(NetPars.stdIntNois, 1) ~= NetPars.numNets)
    error('The dim. of stdIntNois does not match with num. of nets.')
end



