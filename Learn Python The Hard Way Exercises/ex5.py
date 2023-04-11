name = 'Zed A. Shaw'
age = 35 # not a lie
height = 74 # inches
weight = 180 # lbs
eyes = 'Blue'
teeth = 'White'
hair = 'Brown'

inches_to_cm = 1/2.54
lbs_to_kg = 1/2.205

print(f"Let's talk about {name}.")
print(f"He's {round(height*inches_to_cm)} cm tall.")
print(f"He's {round(weight*lbs_to_kg)} kg heavy.")
print("Actually that's not too heavy.")
print(f"He's got {eyes} eyes and {hair} hair.")
print(f"His teeth are usually {teeth} depending on the coffee.")

# this line is tricky, try to get it exactly right
total = age + height*inches_to_cm + weight*lbs_to_kg
print(f"If I add {age}, {round(height*inches_to_cm)}, and {round(weight*lbs_to_kg)} I get {round(total)}.")