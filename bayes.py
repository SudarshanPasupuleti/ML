probAbsentFriday=0.0
probFriday=0.2
#(Absent|Friday)=p(Friday|Absent)p(Absent)/p(Friday)
#p(Friday|Absent)=p(Friday∩Absent)/p(Absent)
#Therefore the result is:
bayesResult=(probAbsentFriday/probFriday)
print(bayesResult * 100)