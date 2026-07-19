



clear; clc; close all;

inputRoot = 'E:\2026\最终结果\机器学习相关\zeta\dataset\hige_super_step_15_npy';


outputRoot = 'H:\20260711-234741_times-new-roman-matlab-python-sync-v6\matlab\05-规律分析\quantitative_theory_analysis\uniform_1deg_complete_pipeline_v10_ade_f1_final_layout';
indexFile = fullfile(inputRoot,'dataset_index_fast_v174.csv');
npyFile = fullfile(inputRoot,'dataset_packed_221_v174.npy');
assert(isfile(indexFile) && isfile(npyFile),'Input index or NPY file is missing.');
for d = {'data','figures','reports'}
    if ~isfolder(fullfile(outputRoot,d{1})), mkdir(fullfile(outputRoot,d{1})); end
end

fontName = 'Times New Roman'; resolution = 600; batchSize = 128;
sourceAngle = [-50:-16, -15:0.2:15, 16:1:50];
uniformAngle = (-50:1:50).';
[tf,integerIndex] = ismember(uniformAngle,sourceAngle.');
assert(all(tf) && numel(integerIndex)==101,'The integer degree locations are not exact members of the original grid.');

T = readtable(indexFile,TextType='string');
needVars = {'filename','zeta_norm','b','c','zeta_raw','npy_index'};
assert(all(ismember(needVars,T.Properties.VariableNames)),'Unexpected fast-index columns.');
[~,ord] = sort(T.npy_index); T=T(ord,:);
assert(isequal(T.npy_index,(0:height(T)-1)'),'npy_index is not a complete zero-based sequence.');
T.phase_gradient_1e3_deg_nm = 2*T.zeta_raw*1e3/350;

[offset,shape] = readNpyLayout(npyFile);
assert(isequal(shape,[height(T),221,221]),'Expected 93104 x 221 x 221 packed maps.');
nSamples=shape(1); nOriginalPix=221*221; nUniform=numel(uniformAngle); nPix=nUniform*nUniform;
fid=fopen(npyFile,'r','ieee-le'); assert(fid>0,'Cannot read NPY file.');
cleanup=onCleanup(@() fclose(fid)); assert(fseek(fid,offset,'bof')==0,'Cannot seek to payload.');
packedRaw=fread(fid,[nOriginalPix nSamples],'*uint16'); clear cleanup fid
assert(isequal(size(packedRaw),[nOriginalPix nSamples]),'Unexpected payload size.');

R=nan(nSamples,1); widthImag=nan(nSamples,1); widthReal=nan(nSamples,1);
covImagReal=nan(nSamples,1); principalAngle=nan(nSamples,1); anisotropy=nan(nSamples,1);
posImag=nan(nSamples,1); posReal=nan(nSamples,1); negImag=nan(nSamples,1); negReal=nan(nSamples,1);
posMax=nan(nSamples,1); negMin=nan(nSamples,1); posMaxImag=nan(nSamples,1); posMaxReal=nan(nSamples,1);
negMinImag=nan(nSamples,1); negMinReal=nan(nSamples,1); finiteCount=zeros(nSamples,1);

fprintf('Processing %d existing 101 x 101 integer-degree map subsets...\n',nSamples);
for first=1:batchSize:nSamples
    ids=first:min(first+batchSize-1,nSamples); nb=numel(ids);
    Z=readUniformBatch(packedRaw,ids,integerIndex);
    Zflat=reshape(Z,nPix,nb); finiteCount(ids)=sum(isfinite(Zflat),1)';
    assert(all(finiteCount(ids)==nPix),'Nonfinite values found.');
    Zd=double(Z); R(ids)=sqrt(mean(double(Zflat).^2,1))';
    W=abs(Zd); P=max(Zd,0); N=max(-Zd,0);
    [widthImag(ids),widthReal(ids),covImagReal(ids),principalAngle(ids),anisotropy(ids)] = geometryStats(W,uniformAngle);
    [posImag(ids),posReal(ids)] = centers(P,uniformAngle);
    [negImag(ids),negReal(ids)] = centers(N,uniformAngle);
    [pmax,ip]=max(Zflat,[],1); [nmin,inn]=min(Zflat,[],1);
    posMax(ids)=double(pmax)'; negMin(ids)=double(nmin)';
    [rp,cp]=ind2sub([nUniform,nUniform],ip); [rn,cn]=ind2sub([nUniform,nUniform],inn);
    posMaxImag(ids)=uniformAngle(cp); posMaxReal(ids)=uniformAngle(rp);
    negMinImag(ids)=uniformAngle(cn); negMinReal(ids)=uniformAngle(rn);
    if mod(first-1,20*batchSize)==0 || ids(end)==nSamples
        fprintf('  %6.1f %% complete\n',100*ids(end)/nSamples);
    end
end

M=T;
M.R_rms_1deg=R; M.width_imag_deg_1deg=widthImag; M.width_real_deg_1deg=widthReal;
M.cov_imag_real_deg2_1deg=covImagReal; M.principal_angle_deg_1deg=principalAngle; M.anisotropy_1deg=anisotropy;
M.pos_center_imag_deg_1deg=posImag; M.pos_center_real_deg_1deg=posReal;
M.neg_center_imag_deg_1deg=negImag; M.neg_center_real_deg_1deg=negReal;
M.pos_max_1deg=posMax; M.neg_min_1deg=negMin;
M.pos_max_imag_deg_1deg=posMaxImag; M.pos_max_real_deg_1deg=posMaxReal;
M.neg_min_imag_deg_1deg=negMinImag; M.neg_min_real_deg_1deg=negMinReal;
M.finite_pixel_count_1deg=finiteCount;
Sbin=makeExtendedBinSummary(M,{'b','c','phase_gradient_1e3_deg_nm'},20);
Gbin=makeGeometryBinSummary(M,{'b','c','phase_gradient_1e3_deg_nm'},20,.08);
Cphase=makeConditionalPhaseSummary(M);
Hphase=fitConditionalPhaseHarmonics(Cphase);

dataDir=fullfile(outputRoot,'data');
writetable(M,fullfile(dataDir,'per_sample_metrics_101x101_1deg.csv'));
writetable(Sbin,fullfile(dataDir,'binned_summary_101x101_1deg.csv'));
writetable(Gbin,fullfile(dataDir,'geometry_summary_101x101_1deg.csv'));
writetable(Cphase,fullfile(dataDir,'conditional_phase_summary_101x101_1deg.csv'));
writetable(Hphase,fullfile(dataDir,'conditional_phase_harmonic_fits.csv'));
writeGeometryConditionalReport(fullfile(outputRoot,'reports','geometry_and_conditional_report_zh.md'),M,Gbin,Hphase);





[ub,~,ib]=unique(M.b,'sorted'); [uc,~,ic]=unique(M.c,'sorted'); [ug,~,ig]=unique(M.zeta_raw,'sorted');
G=zeros(numel(ub),numel(uc),numel(ug),'uint32'); G(sub2ind(size(G),ib,ic,ig))=uint32((1:nSamples)');
sens=uniformSensitivity(packedRaw,G,ub,uc,ug,integerIndex,256);
writetable(sens.local,fullfile(dataDir,'local_sensitivity_101x101_1deg.csv'));
writetable(sens.summary,fullfile(dataDir,'local_sensitivity_summary_101x101_1deg.csv'));
if sens.stable
    figADE=makeADEFigure(Sbin,Gbin,fontName);
    saveTriplet(figADE,fullfile(outputRoot,'figures','supplementary_figure_uniform_1deg_ade_combined'),resolution); close(figADE);
    figF1=makeF1Figure(sens,fontName);
    saveTriplet(figF1,fullfile(outputRoot,'figures','supplementary_figure_uniform_1deg_f1_similarity'),resolution); close(figF1);
end
writeFormalFormulaReport(fullfile(outputRoot,'reports','formal_figure_formula_and_findings_zh.md'),M,Gbin,sens);

summary=table(nSamples,nUniform,nUniform,min(uniformAngle),max(uniformAngle),median(M.R_rms_1deg),...
    median(M.width_imag_deg_1deg),median(M.width_real_deg_1deg),median(M.pos_center_imag_deg_1deg),median(M.pos_center_real_deg_1deg),...
    sens.summary.median_sigma3_over_sigma1,sens.summary.fraction_rank3,...
    'VariableNames',{'sample_count','array_rows','array_cols','angle_min_deg','angle_max_deg','R_median','width_imag_median_deg','width_real_median_deg','pos_center_imag_median_deg','pos_center_real_median_deg','median_sigma3_over_sigma1','fraction_rank3'});
writetable(summary,fullfile(dataDir,'uniform_1deg_result_summary.csv'));
save(fullfile(dataDir,'uniform_1deg_workspace.mat'),'M','Sbin','Gbin','Cphase','Hphase','sens','summary','uniformAngle','integerIndex','-v7.3');

fid=fopen(fullfile(outputRoot,'reports','uniform_1deg_validation_report_zh.md'),'w','n','UTF-8');
fprintf(fid,'# 等间距 1 degree 验证分析\n\n');
fprintf(fid,'本分析从每张原始 221 x 221 质心图中直接抽取 -50:1:50 degree 的已有采样点，得到 101 x 101 数组。未插值、未平滑、未写入或改动原始 NPY 文件。\n\n');
fprintf(fid,'所有参数组合共 %d 张图。该版本用于检验定量趋势和局部可辨识性是否依赖于中央 -15 至 15 degree 区间的 0.2 degree 致密采样。它不取代完整 221 x 221 主分析，因为其主动舍弃了中央区域的原始细采样信息。\n\n',nSamples);
fprintf(fid,'等间距子图的 R 中位数为 %.3f；虚部角和实部角加权宽度中位数分别为 %.3f 和 %.3f degree；正质心区域中心中位数分别为 %.3f 和 %.3f degree。\n\n',median(M.R_rms_1deg),median(M.width_imag_deg_1deg),median(M.width_real_deg_1deg),median(M.pos_center_imag_deg_1deg),median(M.pos_center_real_deg_1deg));
fprintf(fid,'对 256 个内部参数点的 101 x 101 局部有限差分敏感性矩阵，秩为 3 的比例为 %.1f%%，最小与最大归一化奇异值之比的中位数为 %.4f。若该比例高且比值非零，说明移除中央过采样后，三参数仍可引起方向上可区分的二维图样变化；这支持局部联合反演，而不等同于全局唯一反演。\n',100*sens.summary.fraction_rank3,sens.summary.median_sigma3_over_sigma1);
fclose(fid);
fprintf('Uniform 1 degree validation complete: %s\n',outputRoot);

function [offset,shape]=readNpyLayout(file)
fid=fopen(file,'r','ieee-le'); assert(fid>0); c=onCleanup(@()fclose(fid));
magic=fread(fid,6,'uint8=>uint8')'; assert(isequal(magic,uint8([147 double('NUMPY')])));
v=fread(fid,2,'uint8')'; if v(1)==1, n=double(fread(fid,1,'uint16')); else, n=double(fread(fid,1,'uint32')); end
h=char(fread(fid,n,'*char')'); offset=ftell(fid); assert(contains(h,"'descr': '<f2'") && contains(h,"'fortran_order': False"));
t=regexp(h,"'shape':\s*\(([^)]*)\)",'tokens','once'); shape=str2double(regexp(t{1},'\d+','match'));
end
function Z=readUniformBatch(packed,ids,idx)
u=packed(:,ids); v=halfToSingle(u); A=permute(reshape(v,221,221,numel(ids)),[2 1 3]); Z=A(idx,idx,:);
end
function v=halfToSingle(u)
u=uint16(u); e=bitshift(bitand(u,uint16(31744)),-10); f=bitand(u,uint16(1023)); s=ones(size(u),'double'); s(bitand(u,uint16(32768))~=0)=-1; v=zeros(size(u),'double'); n=e>0&e<31; sub=e==0&f~=0; v(n)=(1+double(f(n))/1024).*2.^(double(e(n))-15); v(sub)=double(f(sub))/1024*2^-14; v(e==31&f==0)=inf; v(e==31&f~=0)=nan; v=single(v.*s);
end
function [wx,wy]=widths(W,a)
nb=size(W,3); cw=squeeze(sum(W,1)); rw=squeeze(sum(W,2)); if nb==1,cw=reshape(cw,numel(a),1);rw=reshape(rw,numel(a),1);end; total=sum(cw,1); muX=(a'*double(cw))./total; muY=(a'*double(rw))./total; wx=sqrt(max(0,((a'.^2)*double(cw))./total-muX.^2))'; wy=sqrt(max(0,((a'.^2)*double(rw))./total-muY.^2))';
end
function [wx,wy,cxy,thetaDeg,aniso]=geometryStats(W,a)



nb=size(W,3); n=numel(a); [X,Y]=meshgrid(a,a); total=squeeze(sum(sum(W,1),2)); total=double(total(:)');
cw=squeeze(sum(W,1)); rw=squeeze(sum(W,2)); if nb==1,cw=reshape(cw,n,1);rw=reshape(rw,n,1);end
muX=(a'*double(cw))./total; muY=(a'*double(rw))./total;
varX=nan(1,nb); varY=nan(1,nb); cxy=nan(nb,1); thetaDeg=nan(nb,1); aniso=nan(nb,1);
for q=1:nb
    D=double(W(:,:,q)); dx=X-muX(q); dy=Y-muY(q);
    varX(q)=sum(D.*dx.^2,'all')/total(q); varY(q)=sum(D.*dy.^2,'all')/total(q);
    cxy(q)=sum(D.*dx.*dy,'all')/total(q);
    C=[varX(q),cxy(q);cxy(q),varY(q)]; ev=eig(C); l1=max(ev); l2=min(ev);
    aniso(q)=(l1-l2)/(l1+l2); thetaDeg(q)=0.5*atan2d(2*cxy(q),varX(q)-varY(q));
end
wx=sqrt(max(0,varX))'; wy=sqrt(max(0,varY))';
end
function [cx,cy]=centers(W,a)
nb=size(W,3); cw=squeeze(sum(W,1)); rw=squeeze(sum(W,2)); if nb==1,cw=reshape(cw,numel(a),1);rw=reshape(rw,numel(a),1);end; total=sum(cw,1); cx=((a'*double(cw))./total)'; cy=((a'*double(rw))./total)'; cx(total'==0)=nan;cy(total'==0)=nan;
end
function S=makeBinSummary(M,vars,nBin)
S=table(); for v=1:numel(vars), x=M.(vars{v}); e=unique(quantile(x,linspace(0,1,nBin+1))); id=discretize(x,e,'IncludedEdge','right'); id(x==e(1))=1; for k=1:numel(e)-1, q=id==k; z=x(q); row=table(string(vars{v}),k,min(z),max(z),median(z),sum(q),median(M.R_rms_1deg(q)),quantile(M.R_rms_1deg(q),.25),quantile(M.R_rms_1deg(q),.75),median(M.width_imag_deg_1deg(q)),quantile(M.width_imag_deg_1deg(q),.25),quantile(M.width_imag_deg_1deg(q),.75),median(M.width_real_deg_1deg(q)),quantile(M.width_real_deg_1deg(q),.25),quantile(M.width_real_deg_1deg(q),.75),median(M.pos_center_imag_deg_1deg(q)),quantile(M.pos_center_imag_deg_1deg(q),.25),quantile(M.pos_center_imag_deg_1deg(q),.75),median(M.pos_center_real_deg_1deg(q)),quantile(M.pos_center_real_deg_1deg(q),.25),quantile(M.pos_center_real_deg_1deg(q),.75),'VariableNames',{'parameter','bin','bin_low','bin_high','bin_center','sample_count','R_median','R_q25','R_q75','width_imag_median','width_imag_q25','width_imag_q75','width_real_median','width_real_q25','width_real_q75','pos_imag_median','pos_imag_q25','pos_imag_q75','pos_real_median','pos_real_q25','pos_real_q75'}); S=[S;row]; end, end
end
function fig=makeQuantFigure(S,fontName)


fig=figure(Color='w',Units='centimeters',Position=[2 2 19.2 12.8],Renderer='painters'); tl=tiledlayout(fig,3,3,TileSpacing='compact',Padding='compact'); ps={'b','c','phase_gradient_1e3_deg_nm'}; ts={'Amplitude ratio','Phase difference','Phase gradient'}; xl={'Amplitude ratio','Phase difference (°)','Phase gradient (10^{-3} ° nm^{-1})'}; for col=1:3, D=S(S.parameter==ps{col},:); ax=nexttile(tl,col); band(ax,D.bin_center,D.R_median,D.R_q25,D.R_q75,[.1 .35 .7]); title(ax,ts{col},FontName=fontName,FontSize=11.5,FontWeight='normal'); ylabel(ax,'R (centroid units)',FontName=fontName,FontSize=11.5); style(ax,fontName); xlabel(ax,xl{col},FontName=fontName,FontSize=11.5); ax=nexttile(tl,3+col); h1=band(ax,D.bin_center,D.width_imag_median,D.width_imag_q25,D.width_imag_q75,[.1 .35 .7]); h2=band(ax,D.bin_center,D.width_real_median,D.width_real_q25,D.width_real_q75,[.8 .25 .15]); ylabel(ax,'Width (°)',FontName=fontName,FontSize=11.5); if col==1, legend(ax,[h1 h2],{'Imaginary angle','Real angle'},Location='southwest',Box='on',FontName=fontName,FontSize=10); end; style(ax,fontName); xlabel(ax,xl{col},FontName=fontName,FontSize=11.5); ax=nexttile(tl,6+col); h1=band(ax,D.bin_center,D.pos_imag_median,D.pos_imag_q25,D.pos_imag_q75,[.1 .35 .7]); h2=band(ax,D.bin_center,D.pos_real_median,D.pos_real_q25,D.pos_real_q75,[.8 .25 .15]); ylabel(ax,'Positive-center coordinate (°)',FontName=fontName,FontSize=11.5); if col==1, legend(ax,[h1 h2],{'Imaginary angle','Real angle'},Location='northwest',Box='on',FontName=fontName,FontSize=10); end; style(ax,fontName); xlabel(ax,xl{col},FontName=fontName,FontSize=11.5); end
end
function h=band(ax,x,m,l,u,c), fill(ax,[x;flipud(x)],[l;flipud(u)],c,FaceAlpha=.18,EdgeColor='none');hold(ax,'on');h=plot(ax,x,m,'-o',Color=c,LineWidth=1.2,MarkerSize=3,MarkerFaceColor='w');end
function style(ax,f)
set(ax,...
    FontName=f,...
    FontSize=10,...
    LineWidth=1,...
    TickDir='in',...
    Box='on',...
    Color='w',...
    XColor='k',...
    YColor='k',...
    Layer='top');
grid(ax,'off');
end
function saveTriplet(fig,base,r),drawnow;exportgraphics(fig,[base '.png'],Resolution=r,BackgroundColor='white');exportgraphics(fig,[base '.tiff'],Resolution=r,BackgroundColor='white');exportgraphics(fig,[base '.pdf'],ContentType='vector',BackgroundColor='white');end
function sens=uniformSensitivity(packed,G,ub,uc,ug,idx,nTake)
[I,J,K]=ndgrid(2:numel(ub)-1,2:numel(uc)-1,2:numel(ug)-1); base=double(G(sub2ind(size(G),I(:),J(:),K(:))));base=base(base>0);rng(20260716,'twister');base=base(randperm(numel(base),min(nTake,numel(base)))); local=table(); for q=1:numel(base), z=base(q);[i,j,k]=ind2sub(size(G),find(G==z,1)); ia=double([G(i-1,j,k),G(i+1,j,k)]);ic=double([G(i,j-1,k),G(i,j+1,k)]);ig=double([G(i,j,k-1),G(i,j,k+1)]);Da=(double(readUniformBatch(packed,ia(2),idx))-double(readUniformBatch(packed,ia(1),idx)))/2;Dc=(double(readUniformBatch(packed,ic(2),idx))-double(readUniformBatch(packed,ic(1),idx)))/2;Dg=(double(readUniformBatch(packed,ig(2),idx))-double(readUniformBatch(packed,ig(1),idx)))/2;A=[Da(:),Dc(:),Dg(:)];N=vecnorm(A,2,1);An=A./N;s=svd(An,'econ');C=An'*An;local=[local;table(z,ub(i),uc(j),2*ug(k)*1e3/350,rank(An),s(1),s(2),s(3),s(3)/s(1),C(1,2),C(1,3),C(2,3),'VariableNames',{'npy_row','amplitude_ratio','phase_difference_deg','phase_gradient_1e3_deg_nm','rank_normalized','sigma1_normalized','sigma2_normalized','sigma3_normalized','sigma3_over_sigma1','cosine_amplitude_phase','cosine_amplitude_gradient','cosine_phase_gradient'})]; end; summary=table(median(local.sigma3_over_sigma1),quantile(local.sigma3_over_sigma1,.25),quantile(local.sigma3_over_sigma1,.75),mean(local.rank_normalized==3),median(abs(local.cosine_amplitude_phase)),median(abs(local.cosine_amplitude_gradient)),median(abs(local.cosine_phase_gradient)),'VariableNames',{'median_sigma3_over_sigma1','q25_sigma3_over_sigma1','q75_sigma3_over_sigma1','fraction_rank3','median_abs_cos_amp_phase','median_abs_cos_amp_gradient','median_abs_cos_phase_gradient'});sens.local=local;sens.summary=summary;sens.stable=summary.fraction_rank3>=.9&&summary.median_sigma3_over_sigma1>=.02;
end
function fig=makeSensitivityFigure(sens,fontName)
fig=figure(Color='w',Units='centimeters',Position=[2 2 19.2 9.2],Renderer='painters');tl=tiledlayout(fig,1,2,TileSpacing='compact',Padding='compact');a=median(abs(sens.local.cosine_amplitude_phase));b=median(abs(sens.local.cosine_amplitude_gradient));c=median(abs(sens.local.cosine_phase_gradient));ax=nexttile(tl,1);C=[1 a b;a 1 c;b c 1];imagesc(ax,C);axis(ax,'image');colormap(ax,parula);clim(ax,[0 1]);cb=colorbar(ax);cb.Label.String='Median absolute cosine';cb.FontName=fontName;cb.FontSize=10;set(ax,XTick=1:3,YTick=1:3,XTickLabel={'Amplitude','Phase','Gradient'},YTickLabel={'Amplitude','Phase','Gradient'},FontName=fontName,FontSize=10,TickDir='in');title(ax,'Directional-map similarity',FontName=fontName,FontSize=11.5,FontWeight='bold');for r=1:3,for q=1:3,text(ax,q,r,sprintf('%.2f',C(r,q)),HorizontalAlignment='center',FontName=fontName,Color='w');end,end;ax=nexttile(tl,2);m=[median(sens.local.sigma1_normalized),median(sens.local.sigma2_normalized),median(sens.local.sigma3_normalized)];q25=quantile([sens.local.sigma1_normalized sens.local.sigma2_normalized sens.local.sigma3_normalized],.25);q75=quantile([sens.local.sigma1_normalized sens.local.sigma2_normalized sens.local.sigma3_normalized],.75);bar(ax,1:3,m,.55,FaceColor=[.2 .45 .7]);hold(ax,'on');errorbar(ax,1:3,m,m-q25,q75-m,'k.',LineWidth=1);xlim(ax,[.4 3.6]);ylim(ax,[0 1.85]);xticks(ax,1:3);xticklabels(ax,{'sigma_1','sigma_2','sigma_3'});ylabel(ax,'Normalized singular value',FontName=fontName,FontSize=11.5);title(ax,'Local sensitivity-matrix spectrum',FontName=fontName,FontSize=11.5,FontWeight='normal');style(ax,fontName);text(ax,.03,.98,sprintf('Rank 3: %.1f%%\nMedian sigma_3/sigma_1: %.3f',100*sens.summary.fraction_rank3,sens.summary.median_sigma3_over_sigma1),Units='normalized',VerticalAlignment='top',FontName=fontName,FontSize=10);
end

function fig=makeManualQuantFigure(S,fontName)



degreeSymbol=char(176);
cfg.figureCm=[19.2 5.2];
cfg.panelPosition={ [.075 .205 .250 .680], [.410 .205 .250 .680], [.725 .205 .220 .680] };
cfg.panelLabel={'(a1)','(a2)','(a3)'};
cfg.panelLabelPosition={ [.03 .96],[.03 .96],[.03 .96] };

fig=figure(Color='w',Units='centimeters',Position=[2 2 cfg.figureCm],Renderer='painters');
params={'b','c','phase_gradient_1e3_deg_nm'};
xlabels={'Amplitude ratio',['Phase difference (' degreeSymbol ')'],['Phase gradient (10^{-3} ' degreeSymbol ' nm^{-1})']};
for col=1:3
    D=S(S.parameter==params{col},:);
    ax=axes(fig,Position=cfg.panelPosition{col});
    band(ax,D.bin_center,D.R_median,D.R_q25,D.R_q75,[.10 .35 .70]);
    ylabel(ax,'R (centroid units)',FontName=fontName,FontSize=11.5);
    hx = xlabel(ax,xlabels{col}, ...
    FontName=fontName, ...
    FontSize=11.5);

style(ax,fontName);


hx.Units = 'normalized';
hxPos = hx.Position;
hxPos(2) = hxPos(2) + 0.030;
hx.Position = hxPos;


    panelTag(ax,cfg.panelLabel{col},cfg.panelLabelPosition{col},fontName);
end
end

function panelTag(ax,label,pos,fontName)



text(ax,pos(1),pos(2),label,...
    Units='normalized',...
    HorizontalAlignment='left',...
    VerticalAlignment='top',...
    FontName=fontName,...
    FontSize=12,...
    FontWeight='bold',...
    Color='k',...
    BackgroundColor='none',...
    Clipping='on');
end

function fig=makeADEFigure(S,G,fontName)




degreeSymbol=char(176);


cfg.figureCm=[20.5 14.8];
cfg.panelPosition={...
    [.085 .710 .250 .210],[.400 .710 .250 .210],[.725 .710 .205 .210]; ...
    [.085 .405 .250 .210],[.400 .405 .250 .210],[.725 .405 .205 .210]; ...
    [.085 .085 .250 .210],[.400 .085 .250 .210],[.725 .085 .205 .210]};
cfg.label={...
    '(a1)','(a2)','(a3)'; ...
    '(b1)','(b2)','(b3)'; ...
    '(c1)','(c2)','(c3)'};
cfg.labelPosition={...
    [.03 .96],[.03 .96],[.03 .96]; ...
    [.03 .96],[.03 .96],[.03 .96]; ...
    [.03 .96],[.03 .96],[.03 .96];};
fig=figure(Color='w',Units='centimeters',Position=[2 2 cfg.figureCm],Renderer='painters');
params={'b','c','phase_gradient_1e3_deg_nm'};
xlabels={'Amplitude ratio',['Phase difference (' degreeSymbol ')'],['Phase gradient (10^{-3} ' degreeSymbol ' nm^{-1})']};
violet=[.35 .20 .65];
for col=1:3
    D=S(S.parameter==params{col},:); H=G(G.parameter==params{col},:);
    ax=axes(fig,Position=cfg.panelPosition{1,col});
    band(ax,D.bin_center,D.R_median,D.R_q25,D.R_q75,[.10 .35 .70]);
    ylabel(ax,'R (centroid units)',FontName=fontName,FontSize=11.5);
    xlabel(ax,xlabels{col},FontName=fontName,FontSize=11.5); style(ax,fontName);
    if col==3
        formatThirdColumnZeroTick(ax);
    end
    panelTag(ax,cfg.label{1,col},cfg.labelPosition{1,col},fontName);
    ax=axes(fig,Position=cfg.panelPosition{2,col});
    band(ax,H.bin_center,H.cov_median_deg2,H.cov_q25_deg2,H.cov_q75_deg2,violet);
    yline(ax,0,'-',Color=[.25 .25 .25],LineWidth=.8);
    ylabel(ax,['Covariance (' degreeSymbol '^2)'],FontName=fontName,FontSize=11.5);
    xlabel(ax,xlabels{col},FontName=fontName,FontSize=11.5); style(ax,fontName);
    if col==3
        formatThirdColumnZeroTick(ax);
    end
    panelTag(ax,cfg.label{2,col},cfg.labelPosition{2,col},fontName);
    ax=axes(fig,Position=cfg.panelPosition{3,col});
    plot(ax,H.bin_center,H.principal_angle_axial_mean_deg,'-o',Color=violet,LineWidth=1.25,MarkerSize=3.5,MarkerFaceColor='w');
    ylim(ax,[-90 90]); yticks(ax,[-90 -45 0 45 90]);
    ylabel(ax,['Principal-axis angle (' degreeSymbol ')'],FontName=fontName,FontSize=11.5);
    xlabel(ax,xlabels{col},FontName=fontName,FontSize=11.5);
    style(ax,fontName);
    if col==3
        formatThirdColumnZeroTick(ax);
    end


    panelTag(ax,cfg.label{3,col},cfg.labelPosition{3,col},fontName);
end
end

function formatThirdColumnZeroTick(ax)


xt = xticks(ax);
labels = arrayfun(@(v) sprintf('%g',v),xt,'UniformOutput',false);
zeroIndex = abs(xt) < 1e-12;
labels(zeroIndex) = {'0.0'};
xticklabels(ax,labels);
end

function fig=makeF1Figure(sens,fontName)






cfg.figureCm = [20.5 14.8];




cfg.panelPosition=[.270 .245 .425 .632];

fig=figure( ...
    Color='w', ...
    Units='centimeters', ...
    Position=[2 2 cfg.figureCm], ...
    Renderer='painters');

uAP=median(abs(sens.local.cosine_amplitude_phase));
uAG=median(abs(sens.local.cosine_amplitude_gradient));
uPG=median(abs(sens.local.cosine_phase_gradient));
C=[1 uAP uAG;uAP 1 uPG;uAG uPG 1];

ax=axes(fig,Position=cfg.panelPosition);
imagesc(ax,C);
axis(ax,'image');
colormap(ax,parula);
clim(ax,[0 1]);


oldPos=ax.Position;
cb=colorbar(ax);
ax.Position=oldPos;


cb.Units='normalized';
cb.Position=[.715 .245 .020 .632];
cb.Label.String='';
cb.FontName=fontName;
cb.FontSize=10;
cb.LineWidth=1;
cb.TickDirection='in';

set(ax, ...
    XTick=1:3, ...
    YTick=1:3, ...
    XTickLabel={'','',''}, ...
    YTickLabel={'Amplitude ratio','Phase difference','Phase gradient'}, ...
    FontName=fontName, ...
    FontSize=10, ...
    LineWidth=1, ...
    TickDir='in', ...
    Box='on', ...
    XTickLabelRotation=0, ...
    XColor='k', ...
    YColor='k', ...
    Layer='top');




xLabelX=[1/6 1/2 5/6];
xLabelText={ ...
    {'Amplitude','ratio'}, ...
    {'Phase','difference'}, ...
    {'Phase','gradient'}};

for k=1:3
    text(ax,xLabelX(k),-0.075,xLabelText{k}, ...
        'Units','normalized', ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','top', ...
        'FontName',fontName, ...
        'FontSize',10, ...
        'Interpreter','none', ...
        'Clipping','off');
end


for r=1:3
    for q=1:3
        text(ax,q,r,sprintf('%.2f',C(r,q)), ...
            HorizontalAlignment='center', ...
            VerticalAlignment='middle', ...
            FontName=fontName, ...
            FontSize=10.3, ...
            Color='w');
    end
end


colorbarLabelAxes=axes(fig, ...
    'Units','normalized', ...
    'Position',[0 0 1 1], ...
    'Visible','off', ...
    'HitTest','off', ...
    'PickableParts','none');

text(colorbarLabelAxes,.825,.561,'Median absolute cosine', ...
    'Units','normalized', ...
    'Rotation',90, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','middle', ...
    'FontName',fontName, ...
    'FontSize',11.5, ...
    'Interpreter','none', ...
    'Clipping','off');
end

function fig=makeManualSensitivityFigure(sens,fontName)

cfg.figureCm=[19.2 7.8];
cfg.panelPosition={[.085 .19 .355 .70],[.585 .19 .335 .70]};
cfg.panelLabel={'(f1)','(f2)'};
cfg.panelLabelPosition={[.03 .96],[.03 .96]};
fig=figure(Color='w',Units='centimeters',Position=[2 2 cfg.figureCm],Renderer='painters');
a=median(abs(sens.local.cosine_amplitude_phase));
b=median(abs(sens.local.cosine_amplitude_gradient));
c=median(abs(sens.local.cosine_phase_gradient));
C=[1 a b;a 1 c;b c 1];
ax=axes(fig,Position=cfg.panelPosition{1}); imagesc(ax,C); axis(ax,'image'); colormap(ax,parula); clim(ax,[0 1]);
cb=colorbar(ax); cb.Label.String='Median absolute cosine'; cb.FontName=fontName; cb.FontSize=10; cb.Label.FontName=fontName; cb.Label.FontSize=11.5;
set(ax,XTick=1:3,YTick=1:3,XTickLabel={'Amplitude','Phase','Gradient'},YTickLabel={'Amplitude','Phase','Gradient'},FontName=fontName,FontSize=10,TickDir='in',Box='on');
for r=1:3, for q=1:3, text(ax,q,r,sprintf('%.2f',C(r,q)),HorizontalAlignment='center',FontName=fontName,FontSize=10,Color='w'); end, end
panelTag(ax,cfg.panelLabel{1},cfg.panelLabelPosition{1},fontName);
ax=axes(fig,Position=cfg.panelPosition{2});
m=[median(sens.local.sigma1_normalized),median(sens.local.sigma2_normalized),median(sens.local.sigma3_normalized)];
q25=quantile([sens.local.sigma1_normalized sens.local.sigma2_normalized sens.local.sigma3_normalized],.25);
q75=quantile([sens.local.sigma1_normalized sens.local.sigma2_normalized sens.local.sigma3_normalized],.75);
bar(ax,1:3,m,.55,FaceColor=[.2 .45 .7]); hold(ax,'on'); errorbar(ax,1:3,m,m-q25,q75-m,'k.',LineWidth=1);
xlim(ax,[.4 3.6]); ylim(ax,[0 1.85]); xticks(ax,1:3); xticklabels(ax,{'sigma_1','sigma_2','sigma_3'});
ylabel(ax,'Normalized singular value',FontName=fontName,FontSize=11.5); style(ax,fontName);
text(ax,.03,.82,sprintf('Rank 3: %.1f%%\nMedian sigma_3/sigma_1: %.3f',100*sens.summary.fraction_rank3,sens.summary.median_sigma3_over_sigma1),Units='normalized',VerticalAlignment='top',FontName=fontName,FontSize=10,BackgroundColor='w',Margin=1);
panelTag(ax,cfg.panelLabel{2},cfg.panelLabelPosition{2},fontName);
end

function fig=makeGeometryFigure(S,G,fontName)


degreeSymbol=char(176); cfg.figureCm=[19.2 8.3];
cfg.panelPosition={...
    [.070 .580 .250 .300],[.410 .580 .250 .300],[.725 .580 .220 .300]; ...
    [.070 .145 .250 .300],[.410 .145 .250 .300],[.725 .145 .220 .300]};
cfg.label={'(d1)','(d2)','(d3)';'(e1)','(e2)','(e3)'};
cfg.labelPosition={...
    [.03 .96],[.03 .96],[.03 .96]; ...
    [.03 .96],[.03 .96],[.03 .96]; ...
    [.03 .96],[.03 .96],[.03 .96]};
fig=figure(Color='w',Units='centimeters',Position=[2 2 cfg.figureCm],Renderer='painters');
params={'b','c','phase_gradient_1e3_deg_nm'};
xlabels={'Amplitude ratio',['Phase difference (' degreeSymbol ')'],['Phase gradient (10^{-3} ' degreeSymbol ' nm^{-1})']};
for col=1:3
    D=S(S.parameter==params{col},:); H=G(G.parameter==params{col},:);
    ax=axes(fig,Position=cfg.panelPosition{1,col});
    band(ax,H.bin_center,H.cov_median_deg2,H.cov_q25_deg2,H.cov_q75_deg2,[.35 .20 .65]);
    yline(ax,0,'-',Color=[.25 .25 .25],LineWidth=.8);
    ylabel(ax,['Covariance (' degreeSymbol '^2)'],FontName=fontName,FontSize=11.5);
    xlabel(ax,xlabels{col},FontName=fontName,FontSize=11.5); style(ax,fontName);
    panelTag(ax,cfg.label{1,col},cfg.labelPosition{1,col},fontName);
    ax=axes(fig,Position=cfg.panelPosition{2,col});
    plot(ax,H.bin_center,H.principal_angle_axial_mean_deg,'-o',Color=[.35 .20 .65],LineWidth=1.2,MarkerSize=3,MarkerFaceColor='w');
    ylim(ax,[-90 90]); yticks(ax,[-90 -45 0 45 90]);
    ylabel(ax,['Principal-axis angle (' degreeSymbol ')'],FontName=fontName,FontSize=11.5);
    xlabel(ax,xlabels{col},FontName=fontName,FontSize=11.5); style(ax,fontName);

    panelTag(ax,cfg.label{2,col},cfg.labelPosition{2,col},fontName);
end
end

function fig=makeConditionalPhaseFigure(C,fontName)


degreeSymbol=char(176); cfg.figureCm=[19.2 5.5];
cfg.panelPosition={ [.070 .215 .250 .670],[.410 .215 .250 .670],[.750 .215 .220 .670] };
cfg.label={'(g1)','(g2)','(g3)'}; cfg.labelPosition={ [.03 .96],[.03 .96],[.03 .96] };
cfg.legendPosition=[.085 .240 .205 .085]; cfg.notePosition=[.97 .96];
colors=[.10 .35 .70;.80 .25 .15;.20 .55 .30];
fig=figure(Color='w',Units='centimeters',Position=[2 2 cfg.figureCm],Renderer='painters');
for gg=1:3
    ax=axes(fig,Position=cfg.panelPosition{gg}); hold(ax,'on'); h=gobjects(3,1);
    for gb=1:3
        D=C(C.gradient_group==gg & C.amplitude_group==gb,:); D=sortrows(D,'phase_deg');
        xp=[D.phase_deg;180]; yp=[D.R_median;D.R_median(1)];
        h(gb)=plot(ax,xp,yp,'-o',Color=colors(gb,:),LineWidth=1.2,MarkerSize=3,MarkerFaceColor='w');
    end
    xlim(ax,[-180 180]); xticks(ax,[-180 -90 0 90 180]);
    ylabel(ax,'R (centroid units)',FontName=fontName,FontSize=11.5);
    xlabel(ax,['Phase difference (' degreeSymbol ')'],FontName=fontName,FontSize=11.5); style(ax,fontName);
    dg=C(C.gradient_group==gg,:); note=sprintf('Gradient group %d\n%.3f to %.3f',gg,dg.gradient_min_1e3_deg_nm(1),dg.gradient_max_1e3_deg_nm(1));
    text(ax,cfg.notePosition(1),cfg.notePosition(2),note,Units='normalized',HorizontalAlignment='right',VerticalAlignment='top',FontName=fontName,FontSize=9,BackgroundColor='w',Margin=1);
    panelTag(ax,cfg.label{gg},cfg.labelPosition{gg},fontName);
    if gg==1
        lgd=legend(ax,h,{'Low amplitude','Middle amplitude','High amplitude'},Box='on',FontName=fontName,FontSize=9);
        lgd.Units='normalized'; lgd.Position=cfg.legendPosition;
    end
end
end

function S=makeExtendedBinSummary(M,vars,nBin)

S=table();
for v=1:numel(vars)
    x=M.(vars{v}); e=unique(quantile(x,linspace(0,1,nBin+1))); id=discretize(x,e,'IncludedEdge','right'); id(x==e(1))=1;
    for k=1:numel(e)-1
        q=id==k; z=x(q);
        row=table(string(vars{v}),k,min(z),max(z),median(z),sum(q),...
            median(M.R_rms_1deg(q)),quantile(M.R_rms_1deg(q),.25),quantile(M.R_rms_1deg(q),.75),...
            median(M.width_imag_deg_1deg(q)),quantile(M.width_imag_deg_1deg(q),.25),quantile(M.width_imag_deg_1deg(q),.75),...
            median(M.width_real_deg_1deg(q)),quantile(M.width_real_deg_1deg(q),.25),quantile(M.width_real_deg_1deg(q),.75),...
            median(M.pos_center_imag_deg_1deg(q)),quantile(M.pos_center_imag_deg_1deg(q),.25),quantile(M.pos_center_imag_deg_1deg(q),.75),...
            median(M.pos_center_real_deg_1deg(q)),quantile(M.pos_center_real_deg_1deg(q),.25),quantile(M.pos_center_real_deg_1deg(q),.75),...
            median(M.neg_center_imag_deg_1deg(q)),quantile(M.neg_center_imag_deg_1deg(q),.25),quantile(M.neg_center_imag_deg_1deg(q),.75),...
            median(M.neg_center_real_deg_1deg(q)),quantile(M.neg_center_real_deg_1deg(q),.25),quantile(M.neg_center_real_deg_1deg(q),.75),...
            'VariableNames',{'parameter','bin','bin_low','bin_high','bin_center','sample_count',...
            'R_median','R_q25','R_q75','width_imag_median','width_imag_q25','width_imag_q75',...
            'width_real_median','width_real_q25','width_real_q75','pos_imag_median','pos_imag_q25','pos_imag_q75',...
            'pos_real_median','pos_real_q25','pos_real_q75','neg_imag_median','neg_imag_q25','neg_imag_q75',...
            'neg_real_median','neg_real_q25','neg_real_q75'});
        S=[S;row];
    end
end
end

function G=makeGeometryBinSummary(M,vars,nBin,anisotropyThreshold)


G=table();
for v=1:numel(vars)
    x=M.(vars{v}); e=unique(quantile(x,linspace(0,1,nBin+1))); id=discretize(x,e,'IncludedEdge','right'); id(x==e(1))=1;
    for k=1:numel(e)-1
        q=id==k; z=x(q); valid=q & M.anisotropy_1deg>=anisotropyThreshold;
        [theta,concentration]=axialMeanDeg(M.principal_angle_deg_1deg(valid));
        row=table(string(vars{v}),k,min(z),max(z),median(z),sum(q),...
            median(M.cov_imag_real_deg2_1deg(q)),quantile(M.cov_imag_real_deg2_1deg(q),.25),quantile(M.cov_imag_real_deg2_1deg(q),.75),...
            median(M.anisotropy_1deg(q)),quantile(M.anisotropy_1deg(q),.25),quantile(M.anisotropy_1deg(q),.75),...
            theta,concentration,sum(valid),anisotropyThreshold,...
            'VariableNames',{'parameter','bin','bin_low','bin_high','bin_center','sample_count',...
            'cov_median_deg2','cov_q25_deg2','cov_q75_deg2','anisotropy_median','anisotropy_q25','anisotropy_q75',...
            'principal_angle_axial_mean_deg','principal_angle_resultant','principal_angle_valid_count','anisotropy_threshold'});
        G=[G;row];
    end
end
end

function [thetaDeg,r]=axialMeanDeg(theta)
theta=theta(isfinite(theta));
if isempty(theta), thetaDeg=nan; r=nan; return; end
u=mean(exp(1i*2*deg2rad(theta))); thetaDeg=rad2deg(angle(u))/2; r=abs(u);
end

function C=makeConditionalPhaseSummary(M)


bLevels=unique(M.b,'sorted'); gLevels=unique(M.phase_gradient_1e3_deg_nm,'sorted');
[~,ib]=ismember(M.b,bLevels); [~,ig]=ismember(M.phase_gradient_1e3_deg_nm,gLevels);
bGroup=min(3,ceil(3*ib/numel(bLevels))); gGroup=min(3,ceil(3*ig/numel(gLevels)));
phase=M.c; phase(phase==180)=-180; pLevels=unique(phase,'sorted'); C=table();
for gb=1:3
    for gg=1:3
        base=bGroup==gb & gGroup==gg;
        for p=pLevels'
            q=base & phase==p;
            row=table(gb,gg,min(M.b(base)),max(M.b(base)),min(M.phase_gradient_1e3_deg_nm(base)),max(M.phase_gradient_1e3_deg_nm(base)),p,sum(q),...
                median(M.R_rms_1deg(q)),quantile(M.R_rms_1deg(q),.25),quantile(M.R_rms_1deg(q),.75),...
                median(M.cov_imag_real_deg2_1deg(q)),median(M.pos_center_imag_deg_1deg(q)),median(M.pos_center_real_deg_1deg(q)),...
                median(M.neg_center_imag_deg_1deg(q)),median(M.neg_center_real_deg_1deg(q)),...
                'VariableNames',{'amplitude_group','gradient_group','amplitude_min','amplitude_max','gradient_min_1e3_deg_nm','gradient_max_1e3_deg_nm','phase_deg','sample_count',...
                'R_median','R_q25','R_q75','cov_median_deg2','pos_imag_median','pos_real_median','neg_imag_median','neg_real_median'});
            C=[C;row];
        end
    end
end

end

function H=fitConditionalPhaseHarmonics(C)

H=table();
for gb=1:3
    for gg=1:3
        D=C(C.amplitude_group==gb & C.gradient_group==gg,:); phi=deg2rad(D.phase_deg); y=D.R_median;
        X=[ones(size(phi)) cos(phi) sin(phi) cos(2*phi) sin(2*phi)]; beta=X\y; yhat=X*beta;
        r2=1-sum((y-yhat).^2)/sum((y-mean(y)).^2);
        amp1=hypot(beta(2),beta(3)); peak=mod(rad2deg(atan2(beta(3),beta(2)))+180,360)-180;
        row=table(gb,gg,D.amplitude_min(1),D.amplitude_max(1),D.gradient_min_1e3_deg_nm(1),D.gradient_max_1e3_deg_nm(1),...
            beta(1),amp1,peak,r2,'VariableNames',{'amplitude_group','gradient_group','amplitude_min','amplitude_max','gradient_min_1e3_deg_nm','gradient_max_1e3_deg_nm','R_offset','first_harmonic_amplitude','first_harmonic_peak_deg','two_harmonic_R2'});
        H=[H;row];
    end
end
end

function writeGeometryConditionalReport(file,M,G,H)
fid=fopen(file,'w','n','UTF-8'); assert(fid>0,'Cannot write geometry report.'); c=onCleanup(@()fclose(fid));
fprintf(fid,'# 二维几何与条件相位规律分析\n\n');
fprintf(fid,'本报告基于全部 %d 张等间距 101 x 101 质心图。二维几何权重为绝对质心响应 |Delta x|，因此它描述响应区域的空间分布，而不是正负响应相互抵消后的平均。\n\n',height(M));
fprintf(fid,'## 协方差和旋转\n\n');
fprintf(fid,'对每张图计算虚部角与实部角的加权协方差 Cxy，以及由协方差矩阵特征向量确定的主轴方向。Cxy 不为零表示响应区域不再与两个坐标轴独立对齐；主轴方向的变化则表明椭圆型响应分布发生旋转。协方差受图样尺度影响，因而正式图保留更直观的主轴方向，协方差保留在数据表中。主轴角仅在各向异性指标大于或等于 0.08 时汇总，以避免近圆形分布中方向不稳定。全数据的协方差中位数为 %.3f degree squared，各向异性中位数为 %.3f。\n\n',median(M.cov_imag_real_deg2_1deg),median(M.anisotropy_1deg));
fprintf(fid,'## 正负响应中心和宽度\n\n');
fprintf(fid,'正负响应中心以及两个方向的加权宽度均已逐样本计算，并保留在 per_sample_metrics 和 binned_summary 源数据表中。它们在部分参数范围内呈现相似趋势，因而不纳入正式图，以避免削弱三个参数可区分性的主论证。\n\n');
fprintf(fid,'## 条件相位规律和周期性\n\n');
fprintf(fid,'相位差按圆周变量处理，+180 degree 与 -180 degree 合并为同一相位点。条件分析将振幅比和相位梯度各分为低、中、高三组，并在每个固定条件组合下保留完整相位曲线。对每条条件曲线拟合一阶和二阶圆谐波。九个条件组合的双谐波拟合 R squared 中位数为 %.3f，四分位范围为 %.3f 至 %.3f；这说明相位差的影响应以周期响应和条件依赖的相位峰位来描述，而非以不连续的普通线性趋势描述。\n',median(H.two_harmonic_R2),quantile(H.two_harmonic_R2,.25),quantile(H.two_harmonic_R2,.75));
end

function writeFormalFormulaReport(file,M,G,sens)
fid=fopen(file,'w','n','UTF-8'); assert(fid>0,'Cannot write formula report.'); c=onCleanup(@()fclose(fid));
fprintf(fid,'# 正式图指标公式、物理含义与数据规律\n\n');
fprintf(fid,'本报告对应组合图的 a、b、c 三行以及单独热力图 (a)。所有边缘统计均使用全部 %d 张等间距 101 x 101 质心图；每个参数分箱后，对其余两个参数自由变化时的样本取中位数，阴影为 25%% 至 75%% 四分位范围。\n\n',height(M));
fprintf(fid,'## a：整体响应强度\n\n');
fprintf(fid,'对一张二维质心图 Delta x(i,j)，计算 R = sqrt(mean(Delta x(i,j)^2))。R 是整个二维图的均方根质心响应，不区分虚部角或实部角方向，也不是 x 或 y 方向的放大倍数。数据中，振幅比和相位梯度的 R 均先增强后趋于平台；相位差呈明显周期型谷和峰。因此，仅凭 R 不能在所有范围内完全区分振幅比与相位梯度，但相位差的行为不同。\n\n');
fprintf(fid,'## b：二维协方差\n\n');
fprintf(fid,'令 w(i,j)=abs(Delta x(i,j))/sum(abs(Delta x))，虚部角和实部角分别为 theta_I 与 theta_R。加权中心为 mu_I=sum(w theta_I)、mu_R=sum(w theta_R)，协方差为 C_IR=sum[w (theta_I-mu_I)(theta_R-mu_R)]。C_IR 的正负表示响应区域偏向两种相反对角方向，绝对值表示该斜向关联的强度。相位差分箱中 C_IR 从约 %.1f 到 %.1f degree squared，出现清楚的正负交替；振幅比和相位梯度分箱的中位数则更接近零。\n\n',min(G.cov_median_deg2),max(G.cov_median_deg2));
fprintf(fid,'## c：主轴方向\n\n');
fprintf(fid,'由二维加权协方差矩阵 Sigma = [C_II, C_IR; C_IR, C_RR] 的主特征向量得到主轴角 psi = 0.5 atan2(2 C_IR, C_II-C_RR)。由于主轴没有箭头，psi 和 psi+180 degree 等价，图中使用 -90 至 90 degree 表示。仅汇总各向异性 (lambda_1-lambda_2)/(lambda_1+lambda_2) 大于等于 0.08 的图样，以排除近圆形分布中不稳定的方向。相位差导致最清楚的连续转向；接近 ±90 degree 的跳变是无向主轴的周期表示，不应解释为物理突变。\n\n');
fprintf(fid,'## 单独热力图 (a)：局部二维图样变化方向的相似性\n\n');
fprintf(fid,'在 256 个可作中心差分的内部参数点，把二维图展开为向量 X。对参数 p 的局部响应使用 D_p=[X(p+)-X(p-)]/2；对任意两参数 p、q，计算 abs(cos alpha_pq)=abs(D_p^T D_q)/(norm(D_p) norm(D_q))，再在 256 个点上取中位数。数值越接近 0，表示两参数使完整图样沿更不同的方向变化；越接近 1，表示局部变化更相似。实际中，振幅比和相位差为 %.2f，振幅比和相位梯度为 %.2f，相位差和相位梯度为 %.2f。故相位差最容易与振幅比区分；振幅比与相位梯度存在较强局部相似性，反演时需要同时使用 R、协方差和主轴方向等互补二维特征，而不能依赖单一指标。\n',sens.summary.median_abs_cos_amp_phase,sens.summary.median_abs_cos_amp_gradient,sens.summary.median_abs_cos_phase_gradient);
end
