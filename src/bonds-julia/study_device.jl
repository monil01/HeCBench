# Source-aligned FP32 dated cashflows / safeguarded yield search. Source attribution retained.
include(joinpath(@__DIR__,"study_types.jl"))
include(joinpath(@__DIR__,"study_date_tables.jl"))
@inline function bdate(d,m,y)
 leap=YEARLEAP[y-1900+1]!=0;offset=leap ? MONTHLEAPOFFSET[m] : MONTHOFFSET[m]
 BDate(Int32(m),Int32(d),Int32(y),Int32(d+offset+YEAROFFSET[y-1900+1]))
end
@inline function advance(date,months)
 d=date.day;m=date.month+months;y=date.year
 while m>12;m-=12;y+=1;end
 while m<1;m+=12;y-=1;end
 length=YEARLEAP[y-1900+1]!=0 ? MONTHLEAPLEN[m] : MONTHLEN[m]
 if d>length;d=length;end
 bdate(d,m,y)
end
@inline function daycount(a,b,dc)
 if dc==0
  d1=a.day;d2=b.day;m1=a.month;m2=b.month
  if d2==31 && d1<30;d2=1;m2+=1;end
  Int32(360)*(b.year-a.year)+Int32(30)*(m2-m1-Int32(1))+max(Int32(0),Int32(30)-d1)+min(Int32(30),d2)
 else;b.serial-a.serial
 end
end
@inline fraction(a,b,dc)=Float32(daycount(a,b,dc))/360f0
@inline function compound(y,t)
 if y.comp==0;1f0+y.rate*t
 elseif y.comp==1;(1f0+y.rate/y.freq)^(y.freq*t)
 elseif y.comp==2;exp(y.rate*t)
 else;0f0
 end
end
@inline discount(y,t)=1f0/compound(y,t)
@inline function amount(legs,j,couponrate)
 c=legs[j]
 c.amount== -1f0 ? 100f0*(compound(BRate(couponrate,1f0,Int32(0),Int32(0)),fraction(c.start,c.finish,Int32(0)))-1f0) : c.amount
end
@inline function accrued(legs,count,couponrate,date,maturity)
 first=count
 for j in 1:count
  if !(date.serial>legs[j].pay.serial);first=j;break;end
 end
 result=0f0
 for j in first:count
  c=legs[j]
  if !(date.serial<=c.start.serial || date.serial>maturity.serial)
   ending=date.serial<c.finish.serial ? date : c.finish
   result+=100f0*(compound(BRate(couponrate,1f0,Int32(0),Int32(0)),fraction(c.start,ending,Int32(0)))-1f0)
  end
 end
 result
end
@inline function npv_yield(legs,count,couponrate,y,date)
 npv=0f0;disc=1f0;first=true;last=date
 for j in 1:count
  c=legs[j]
  if date.serial>c.pay.serial;continue;end
  if first
   first=false
   last=j>1 ? advance(c.pay,Int32(-6)) : c.start
   disc*=discount(y,fraction(date,c.pay,y.dc))
  else;disc*=discount(y,fraction(last,c.pay,y.dc))
  end
  last=c.pay;npv+=amount(legs,j,couponrate)*disc
 end
 npv
end
@inline function duration(legs,count,couponrate,y,date)
 P=0f0;deriv=0f0;r=y.rate;N=y.freq
 for j in 1:count
  c=legs[j]
  if !(date.serial>c.pay.serial)
   t=fraction(date,c.pay,y.dc);cash=amount(legs,j,couponrate);B=discount(y,t);P+=cash*B
   if y.comp==0;deriv-=cash*B*B*t;end
   if y.comp==1;deriv-=cash*t*B/(1+r/N);end
   if y.comp==2;deriv-=cash*B*t;end
   if y.comp==3
    if t<=1f0/N;deriv-=cash*B*B*t
    else;deriv-=cash*t*B/(1f0+r/N)
    end
   end
  end
 end
 P==0f0 ? 0f0 : (-1*deriv)/P
end
@inline objective(target,y,legs,count,couponrate,date)=target-npv_yield(legs,count,couponrate,BRate(y,2f0,Int32(1),Int32(0)),date)
@inline derivative(y,legs,count,couponrate,date)=duration(legs,count,couponrate,BRate(y,2f0,Int32(1),Int32(0)),date)
@inline function close_source(x,y)
 diff=abs(x-y);tol=42*1f-18
 diff<=tol*abs(x) && diff<=tol*abs(y)
end
@inline function solve_impl(root,xmin,xmax,fmin,evaluations,target,legs,count,couponrate,date,accuracy)
 if fmin<0f0;xl=xmin;xh=xmax
 else;xh=xmin;xl=xmax
 end
 old=xmax-xmin;dx=old
 froot=objective(target,root,legs,count,couponrate,date);df=derivative(root,legs,count,couponrate,date);evaluations+=1
 while evaluations<=100
  if ((root-xh)*df-froot)*((root-xl)*df-froot)>0f0 || abs(2f0*froot)>abs(old*df)
   old=dx;dx=(xh-xl)/2f0;root=xl+dx
  else;old=dx;dx=froot/df;root-=dx
  end
  if abs(dx)<accuracy;return root;end
  froot=objective(target,root,legs,count,couponrate,date);df=derivative(root,legs,count,couponrate,date);evaluations+=1
  if froot<0f0;xl=root
  else;xh=root
  end
 end
 root
end
@inline function solve_yield(target,legs,count,couponrate,date)
 accuracy=max(Float32(1e-8),1f-18);growth=1.6f0;flip=-1;root=.05f0;step=root/10f0
 fmax=objective(target,root,legs,count,couponrate,date)
 if close_source(fmax,0f0);return root
 elseif close_source(fmax,0f0)
  xmin=root-step;fmin=objective(target,xmin,legs,count,couponrate,date);xmax=root
 else;xmin=root;fmin=fmax;xmax=root+step;fmax=objective(target,xmax,legs,count,couponrate,date)
 end
 evaluations=2
 while evaluations<=100
  if fmin*fmax<=0f0
   if close_source(fmin,0f0);return xmin;end
   if close_source(fmax,0f0);return xmax;end
   root=(xmax+xmin)/2f0
   return solve_impl(root,xmin,xmax,fmin,evaluations,target,legs,count,couponrate,date,accuracy)
  end
  if abs(fmin)<abs(fmax)
   xmin=xmin+growth*(xmin-xmax);fmin=objective(target,xmin,legs,count,couponrate,date)
  elseif abs(fmin)>abs(fmax)
   xmax=xmax+growth*(xmax-xmin);fmax=objective(target,xmax,legs,count,couponrate,date)
  elseif flip== -1
   xmin=xmin+growth*(xmin-xmax);fmin=objective(target,xmin,legs,count,couponrate,date);evaluations+=1;flip=1
  elseif flip==1
   xmax=xmax+growth*(xmax-xmin);fmax=objective(target,xmax,legs,count,couponrate,date);flip= -1
  end
  evaluations+=1
 end
 0f0
end
@inline function bonds_body!(curve,repo,current,maturity,clean,bond,strike,dirty,acc,outclean,yields,n,g)
 if g<n
  j=g+1;b=bond[j];date=current[j];count=Int32(0);flow=b.maturity
  while flow.serial>b.start.serial;count+=Int32(1);flow=advance(flow,Int32(-6));end
  count+=Int32(1)
  # Source requires count<=9; host validates before any launch.
  legs=MVector{9,BCoupon}(undef)
  start=advance(b.maturity,(count-Int32(1))*Int32(-6));ending=advance(start,Int32(6))
  for k in Int32(1):count-Int32(1)
   legs[k]=BCoupon(ending,start,ending,-1f0);start=ending;ending=advance(ending,Int32(6))
  end
  legs[count]=BCoupon(b.maturity,date,date,100f0)
  settle=b.start.serial>date.serial ? b.start : date
  target=clean[j]+accrued(legs,count,b.rate,settle,maturity[j])
  target/=100f0/100f0
  y=solve_yield(target,legs,count,b.rate,settle);yields[j]=y
  c=curve[j];c=BYield(y,c.compounding,c.frequency,c.intRate,c.refDate,c.calDate,c.dc);curve[j]=c
  payrate=BRate(c.forward,c.frequency,Int32(c.compounding),c.intRate.dc);value=0f0
  for k in Int32(1):count
   if !(settle.serial>legs[k].pay.serial)
    value+=amount(legs,k,b.rate)*discount(payrate,fraction(c.refDate,legs[k].pay,c.dc))
   end
  end
  value/=discount(payrate,fraction(c.refDate,settle,c.dc))
  dirty[j]=value*100f0/100f0;acc[j]=accrued(legs,count,b.rate,date,maturity[j]);outclean[j]=dirty[j]-acc[j]
 end
 nothing
end
