import pickle
import matplotlib.pyplot as plt
import pandas as pd


trials = pickle.load(open("trials.p", "rb"))
print("Set breakpoint here")


#for item in trials.trials:
#    args = item["vals"]
#    res = item["result"]["loss"]

itemtuples = [(item["misc"]["vals"]["dilation_group"], item["misc"]["vals"]["use_ref_points"], item["result"]["loss"]) for item in trials.trials]
(dil, ref, loss) = zip(*itemtuples)

#plt.figure()

plt.plot(dil, 'ro')
plt.title('Use_dilation (1 is true, 0 is false)')

#plt.plot(loss)
plt.plot(pd.DataFrame(loss).ewm(span=1).mean())
plt.title('MSE')

#plt.plot(ref, 'g^')
#plt.legend()
plt.show()


print("Set breakpoint here")

print("PRINT BEST TRIALS")
myitems = [(trial["result"]["loss"], str(trial["misc"]["vals"])) for trial in trials.trials]
myitems.sort(key=lambda tup: tup[0])
for item in myitems[:10]:
    print("--------------------------\n")
    print(item)
    print("\n\n")
# TODO: Print training times as well... attemptid["book_time"]

