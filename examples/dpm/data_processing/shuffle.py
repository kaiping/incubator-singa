import random

ratio = 0.81
ratio2 = 0.9
f = open("shard_input", "r")
train = open("nuhs_train", "w")
validate = open("nuhs_validation", "w")
test = open("nuhs_test", "w")

data = f.readlines()

split = int(len(data) * ratio)
print "split point: %d" %split
split2 = int(len(data) * ratio2)
print "split point2: %d" %split2

random.shuffle(data)

train_data = data[:split]
validate_data = data[split:split2]
test_data = data[split2:]

print len(train_data) + len(validate_data) + len(test_data)

for item in train_data:
    train.write(item)
for item in validate_data:
    validate.write(item)
for item in test_data:
    test.write(item)

f.close()
train.close()
validate.close()
test.close()
