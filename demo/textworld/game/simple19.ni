Use MAX_STATIC_DATA of 500000.
When play begins, seed the random-number generator with 1234.

container is a kind of thing.
door is a kind of thing.
object-like is a kind of thing.
supporter is a kind of thing.
food is a kind of object-like.
key is a kind of object-like.
containers are openable, lockable and fixed in place. containers are usually closed.
door is openable and lockable.
object-like is portable.
supporters are fixed in place.
food is edible.
A room has a text called internal name.


The r_1 and the r_0 and the r_3 and the r_2 and the r_4 are rooms.

Understand "laundry place" as r_1.
The internal name of r_1 is "laundry place".
The printed name of r_1 is "-= Laundry Place =-".
The laundry place part 0 is some text that varies. The laundry place part 0 is "You are in a laundry place. The room seems oddly familiar, as though it were only superficially different from the other rooms in the building.

 You can see [if c_0 is locked]a locked[else if c_0 is open]an opened[otherwise]a closed[end if]".
The laundry place part 1 is some text that varies. The laundry place part 1 is " TextWorld limited edition box.[if c_0 is open and there is something in the c_0] The TextWorld limited edition box contains [a list of things in the c_0]. Now that's what I call TextWorld![end if]".
The laundry place part 2 is some text that varies. The laundry place part 2 is "[if c_0 is open and the c_0 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The laundry place part 3 is some text that varies. The laundry place part 3 is " You can see a rack. [if there is something on the s_0]You see [a list of things on the s_0] on the rack.[end if]".
The laundry place part 4 is some text that varies. The laundry place part 4 is "[if there is nothing on the s_0]But the thing hasn't got anything on it.[end if]".
The laundry place part 5 is some text that varies. The laundry place part 5 is "

There is an exit to the west. Don't worry, it is unblocked.".
The description of r_1 is "[laundry place part 0][laundry place part 1][laundry place part 2][laundry place part 3][laundry place part 4][laundry place part 5]".

The r_0 is mapped west of r_1.
Understand "cubicle" as r_0.
The internal name of r_0 is "cubicle".
The printed name of r_0 is "-= Cubicle =-".
The cubicle part 0 is some text that varies. The cubicle part 0 is "You are in a cubicle. An ordinary one.

 You rest your hand against a wall, but you miss the wall and fall onto a stand. The stand is ordinary.[if there is something on the s_1] On the stand you make out [a list of things on the s_1].[end if]".
The cubicle part 1 is some text that varies. The cubicle part 1 is "[if there is nothing on the s_1] But the thing hasn't got anything on it.[end if]".
The cubicle part 2 is some text that varies. The cubicle part 2 is " You make out a chair. [if there is something on the s_2]You see [a list of things on the s_2] on the chair.[end if]".
The cubicle part 3 is some text that varies. The cubicle part 3 is "[if there is nothing on the s_2]Unfortunately, there isn't a thing on it. Aw, here you were, all excited for there to be things on it![end if]".
The cubicle part 4 is some text that varies. The cubicle part 4 is "

 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The cubicle part 5 is some text that varies. The cubicle part 5 is " American hatch leading north. There is an unguarded exit to the east.".
The description of r_0 is "[cubicle part 0][cubicle part 1][cubicle part 2][cubicle part 3][cubicle part 4][cubicle part 5]".

north of r_0 and south of r_2 is a door called d_1.
The r_1 is mapped east of r_0.
Understand "garage" as r_3.
The internal name of r_3 is "garage".
The printed name of r_3 is "-= Garage =-".
The garage part 0 is some text that varies. The garage part 0 is "You find yourself in a garage. A typical one. I guess you better just go and list everything you see here.

 You can make out a shelf. The shelf is ordinary.[if there is something on the s_3] On the shelf you can make out [a list of things on the s_3].[end if]".
The garage part 1 is some text that varies. The garage part 1 is "[if there is nothing on the s_3] However, the shelf, like an empty shelf, has nothing on it. It would have been so cool if there was stuff on the shelf.[end if]".
The garage part 2 is some text that varies. The garage part 2 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The garage part 3 is some text that varies. The garage part 3 is " portal leading west. There is an unguarded exit to the north.".
The description of r_3 is "[garage part 0][garage part 1][garage part 2][garage part 3]".

west of r_3 and east of r_2 is a door called d_0.
The r_4 is mapped north of r_3.
Understand "basement" as r_2.
The internal name of r_2 is "basement".
The printed name of r_2 is "-= Basement =-".
The basement part 0 is some text that varies. The basement part 0 is "I am sorry to announce that you are now in the basement.



 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The basement part 1 is some text that varies. The basement part 1 is " portal leading east. There is [if d_1 is open]an open[otherwise]a closed[end if]".
The basement part 2 is some text that varies. The basement part 2 is " American hatch leading south.".
The description of r_2 is "[basement part 0][basement part 1][basement part 2]".

south of r_2 and north of r_0 is a door called d_1.
east of r_2 and west of r_3 is a door called d_0.
Understand "cookhouse" as r_4.
The internal name of r_4 is "cookhouse".
The printed name of r_4 is "-= Cookhouse =-".
The cookhouse part 0 is some text that varies. The cookhouse part 0 is "You are in a cookhouse. You decide to just list off a complete list of everything you see in the room, because hey, why not?

 You can make out [if c_1 is locked]a locked[else if c_1 is open]an opened[otherwise]a closed[end if]".
The cookhouse part 1 is some text that varies. The cookhouse part 1 is " case.[if c_1 is open and there is something in the c_1] The case contains [a list of things in the c_1].[end if]".
The cookhouse part 2 is some text that varies. The cookhouse part 2 is "[if c_1 is open and the c_1 contains nothing] The case is empty! What a waste of a day![end if]".
The cookhouse part 3 is some text that varies. The cookhouse part 3 is "

There is an exit to the south. Don't worry, it is unguarded.".
The description of r_4 is "[cookhouse part 0][cookhouse part 1][cookhouse part 2][cookhouse part 3]".

The r_3 is mapped south of r_4.

The c_0 and the c_1 are containers.
The c_0 and the c_1 are privately-named.
The d_1 and the d_0 are doors.
The d_1 and the d_0 are privately-named.
The f_0 are foods.
The f_0 are privately-named.
The k_1 and the k_0 are keys.
The k_1 and the k_0 are privately-named.
The o_0 are object-likes.
The o_0 are privately-named.
The r_1 and the r_0 and the r_3 and the r_2 and the r_4 are rooms.
The r_1 and the r_0 and the r_3 and the r_2 and the r_4 are privately-named.
The s_0 and the s_1 and the s_2 and the s_3 are supporters.
The s_0 and the s_1 and the s_2 and the s_3 are privately-named.

The description of d_1 is "it is what it is, a American hatch [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_1 is "American hatch".
Understand "American hatch" as d_1.
Understand "American" as d_1.
Understand "hatch" as d_1.
The d_1 is locked.
The description of d_0 is "it is what it is, a portal [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_0 is "portal".
Understand "portal" as d_0.
The d_0 is closed.
The description of c_0 is "The TextWorld limited edition box looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_0 is "TextWorld limited edition box".
Understand "TextWorld limited edition box" as c_0.
Understand "TextWorld" as c_0.
Understand "limited" as c_0.
Understand "edition" as c_0.
Understand "box" as c_0.
The c_0 is in r_1.
The c_0 is closed.
The description of c_1 is "The case looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_1 is "case".
Understand "case" as c_1.
The c_1 is in r_4.
The c_1 is closed.
The description of f_0 is "that's an ordinary gummy bear!".
The printed name of f_0 is "gummy bear".
Understand "gummy bear" as f_0.
Understand "gummy" as f_0.
Understand "bear" as f_0.
The f_0 is in r_3.
The f_0 is edible.
The description of k_1 is "The TextWorld limited edition key is cold to the touch".
The printed name of k_1 is "TextWorld limited edition key".
Understand "TextWorld limited edition key" as k_1.
Understand "TextWorld" as k_1.
Understand "limited" as k_1.
Understand "edition" as k_1.
Understand "key" as k_1.
The k_1 is in r_1.
The matching key of the c_0 is the k_1.
The description of o_0 is "The shoe is clean.".
The printed name of o_0 is "shoe".
Understand "shoe" as o_0.
The o_0 is in r_3.
The description of s_0 is "The rack is balanced.".
The printed name of s_0 is "rack".
Understand "rack" as s_0.
The s_0 is in r_1.
The description of s_1 is "The stand is stable.".
The printed name of s_1 is "stand".
Understand "stand" as s_1.
The s_1 is in r_0.
The description of s_2 is "The chair is stable.".
The printed name of s_2 is "chair".
Understand "chair" as s_2.
The s_2 is in r_0.
The description of s_3 is "The shelf is unstable.".
The printed name of s_3 is "shelf".
Understand "shelf" as s_3.
The s_3 is in r_3.
The description of k_0 is "The metal of the American passkey is antiqued.".
The printed name of k_0 is "American passkey".
Understand "American passkey" as k_0.
Understand "American" as k_0.
Understand "passkey" as k_0.
The player carries the k_0.
The matching key of the d_1 is the k_0.


The player is in r_2.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "unlock American hatch with American passkey / open American hatch / go south / go east / open TextWorld limited edition box"

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The player is in r_1 and The c_0 is in r_1 and The c_0 is open:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest0 completed is true;

Use scoring. The maximum score is 1.
This is the simpler notify score changes rule:
	If the score is not the last notified score:
		let V be the score - the last notified score;
		if V > 0:
			say "Your score has just gone up by [V in words] ";
		else:
			say "Your score changed by [V in words] ";
		if V >= -1 and V <= 1:
			say "point.";
		else:
			say "points.";
		Now the last notified score is the score;
	if quest0 completed is true:
		end the story finally; [Win]

The simpler notify score changes rule substitutes for the notify score changes rule.

Rule for listing nondescript items:
	stop.

Rule for printing the banner text:
	say "[fixed letter spacing]";
	say "                    ________  ________  __    __  ________        [line break]";
	say "                   |        \|        \|  \  |  \|        \       [line break]";
	say "                    \$$$$$$$$| $$$$$$$$| $$  | $$ \$$$$$$$$       [line break]";
	say "                      | $$   | $$__     \$$\/  $$   | $$          [line break]";
	say "                      | $$   | $$  \     >$$  $$    | $$          [line break]";
	say "                      | $$   | $$$$$    /  $$$$\    | $$          [line break]";
	say "                      | $$   | $$_____ |  $$ \$$\   | $$          [line break]";
	say "                      | $$   | $$     \| $$  | $$   | $$          [line break]";
	say "                       \$$    \$$$$$$$$ \$$   \$$    \$$          [line break]";
	say "              __       __   ______   _______   __        _______  [line break]";
	say "             |  \  _  |  \ /      \ |       \ |  \      |       \ [line break]";
	say "             | $$ / \ | $$|  $$$$$$\| $$$$$$$\| $$      | $$$$$$$\[line break]";
	say "             | $$/  $\| $$| $$  | $$| $$__| $$| $$      | $$  | $$[line break]";
	say "             | $$  $$$\ $$| $$  | $$| $$    $$| $$      | $$  | $$[line break]";
	say "             | $$ $$\$$\$$| $$  | $$| $$$$$$$\| $$      | $$  | $$[line break]";
	say "             | $$$$  \$$$$| $$__/ $$| $$  | $$| $$_____ | $$__/ $$[line break]";
	say "             | $$$    \$$$ \$$    $$| $$  | $$| $$     \| $$    $$[line break]";
	say "              \$$      \$$  \$$$$$$  \$$   \$$ \$$$$$$$$ \$$$$$$$ [line break]";
	say "[variable letter spacing][line break]";
	say "[objective][line break]".

Include Basic Screen Effects by Emily Short.

Rule for printing the player's obituary:
	if story has ended finally:
		center "*** The End ***";
	else:
		center "*** You lost! ***";
	say paragraph break;
	if maximum score is -32768:
		say "You scored a total of [score] point[s], in [turn count] turn[s].";
	else:
		say "You scored [score] out of a possible [maximum score], in [turn count] turn[s].";
	[wait for any key;
	stop game abruptly;]
	rule succeeds.

Carry out requesting the score:
	if maximum score is -32768:
		say "You have so far scored [score] point[s], in [turn count] turn[s].";
	else:
		say "You have so far scored [score] out of a possible [maximum score], in [turn count] turn[s].";
	rule succeeds.

Rule for implicitly taking something (called target):
	if target is fixed in place:
		say "The [target] is fixed in place.";
	otherwise:
		say "You need to take the [target] first.";
		set pronouns from target;
	stop.

Does the player mean doing something:
	if the noun is not nothing and the second noun is nothing and the player's command matches the text printed name of the noun:
		it is likely;
	if the noun is nothing and the second noun is not nothing and the player's command matches the text printed name of the second noun:
		it is likely;
	if the noun is not nothing and the second noun is not nothing and the player's command matches the text printed name of the noun and the player's command matches the text printed name of the second noun:
		it is very likely.  [Handle action with two arguments.]

Printing the content of the room is an activity.
Rule for printing the content of the room:
	let R be the location of the player;
	say "Room contents:[line break]";
	list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the world is an activity.
Rule for printing the content of the world:
	let L be the list of the rooms;
	say "World: [line break]";
	repeat with R running through L:
		say "  [the internal name of R][line break]";
	repeat with R running through L:
		say "[the internal name of R]:[line break]";
		if the list of things in R is empty:
			say "  nothing[line break]";
		otherwise:
			list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the inventory is an activity.
Rule for printing the content of the inventory:
	say "You are carrying: ";
	list the contents of the player, as a sentence, giving inventory information, including all contents;
	say ".".

The print standard inventory rule is not listed in any rulebook.
Carry out taking inventory (this is the new print inventory rule):
	say "You are carrying: ";
	list the contents of the player, as a sentence, giving inventory information, including all contents;
	say ".".

Printing the content of nowhere is an activity.
Rule for printing the content of nowhere:
	say "Nowhere:[line break]";
	let L be the list of the off-stage things;
	repeat with thing running through L:
		say "  [thing][line break]";

Printing the things on the floor is an activity.
Rule for printing the things on the floor:
	let R be the location of the player;
	let L be the list of things in R;
	remove yourself from L;
	remove the list of containers from L;
	remove the list of supporters from L;
	remove the list of doors from L;
	if the number of entries in L is greater than 0:
		say "There is [L with indefinite articles] on the floor.";

After printing the name of something (called target) while
printing the content of the room
or printing the content of the world
or printing the content of the inventory
or printing the content of nowhere:
	follow the property-aggregation rules for the target.

The property-aggregation rules are an object-based rulebook.
The property-aggregation rulebook has a list of text called the tagline.

[At the moment, we only support "open/unlocked", "closed/unlocked" and "closed/locked" for doors and containers.]
[A first property-aggregation rule for an openable open thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable closed thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an lockable unlocked thing (this is the mention unlocked lockable rule):
	add "unlocked" to the tagline.

A property-aggregation rule for an lockable locked thing (this is the mention locked lockable rule):
	add "locked" to the tagline.]

A first property-aggregation rule for an openable lockable open unlocked thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable lockable closed unlocked thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an openable lockable closed locked thing (this is the mention locked openables rule):
	add "locked" to the tagline.

A property-aggregation rule for a lockable thing (called the lockable thing) (this is the mention matching key of lockable rule):
	let X be the matching key of the lockable thing;
	if X is not nothing:
		add "match [X]" to the tagline.

A property-aggregation rule for an edible off-stage thing (this is the mention eaten edible rule):
	add "eaten" to the tagline.

The last property-aggregation rule (this is the print aggregated properties rule):
	if the number of entries in the tagline is greater than 0:
		say " ([tagline])";
		rule succeeds;
	rule fails;

The objective part 0 is some text that varies. The objective part 0 is "Hey, thanks for coming over to the TextWorld today, there is something I need you to do for me. First, it would be good if you could make sure that the American hatch in the basement is unlocked. Then".
The objective part 1 is some text that varies. The objective part 1 is ", ensure that the American hatch is open. Then, try to go south. And then, move east. After that, make sure that the TextWorld limited edition box inside the laundry place is wide open. And if you do ".
The objective part 2 is some text that varies. The objective part 2 is "that, you're the winner!".

An objective is some text that varies. The objective is "[objective part 0][objective part 1][objective part 2]".
Printing the objective is an action applying to nothing.
Carry out printing the objective:
	say "[objective]".

Understand "goal" as printing the objective.

The taking action has an object called previous locale (matched as "from").

Setting action variables for taking:
	now previous locale is the holder of the noun.

Report taking something from the location:
	say "You pick up [the noun] from the ground." instead.

Report taking something:
	say "You take [the noun] from [the previous locale]." instead.

Report dropping something:
	say "You drop [the noun] on the ground." instead.

The print state option is a truth state that varies.
The print state option is usually false.

Turning on the print state option is an action applying to nothing.
Carry out turning on the print state option:
	Now the print state option is true.

Turning off the print state option is an action applying to nothing.
Carry out turning off the print state option:
	Now the print state option is false.

Printing the state is an activity.
Rule for printing the state:
	let R be the location of the player;
	say "Room: [line break] [the internal name of R][line break]";
	[say "[line break]";
	carry out the printing the content of the room activity;]
	say "[line break]";
	carry out the printing the content of the world activity;
	say "[line break]";
	carry out the printing the content of the inventory activity;
	say "[line break]";
	carry out the printing the content of nowhere activity;
	say "[line break]".

Printing the entire state is an action applying to nothing.
Carry out printing the entire state:
	say "-=STATE START=-[line break]";
	carry out the printing the state activity;
	say "[line break]Score:[line break] [score]/[maximum score][line break]";
	say "[line break]Objective:[line break] [objective][line break]";
	say "[line break]Inventory description:[line break]";
	say "  You are carrying: [a list of things carried by the player].[line break]";
	say "[line break]Room description:[line break]";
	try looking;
	say "[line break]-=STATE STOP=-";

Every turn:
	if extra description command option is true:
		say "<description>";
		try looking;
		say "</description>";
	if extra inventory command option is true:
		say "<inventory>";
		try taking inventory;
		say "</inventory>";
	if extra score command option is true:
		say "<score>[line break][score][line break]</score>";
	if extra score command option is true:
		say "<moves>[line break][turn count][line break]</moves>";
	if print state option is true:
		try printing the entire state;

When play ends:
	if print state option is true:
		try printing the entire state;

After looking:
	carry out the printing the things on the floor activity.

Understand "print_state" as printing the entire state.
Understand "enable print state option" as turning on the print state option.
Understand "disable print state option" as turning off the print state option.

Before going through a closed door (called the blocking door):
	say "You have to open the [blocking door] first.";
	stop.

Before opening a locked door (called the locked door):
	let X be the matching key of the locked door;
	if X is nothing:
		say "The [locked door] is welded shut.";
	otherwise:
		say "You have to unlock the [locked door] with the [X] first.";
	stop.

Before opening a locked container (called the locked container):
	let X be the matching key of the locked container;
	if X is nothing:
		say "The [locked container] is welded shut.";
	otherwise:
		say "You have to unlock the [locked container] with the [X] first.";
	stop.

Displaying help message is an action applying to nothing.
Carry out displaying help message:
	say "[fixed letter spacing]Available commands:[line break]";
	say "  look:                describe the current room[line break]";
	say "  goal:                print the goal of this game[line break]";
	say "  inventory:           print player's inventory[line break]";
	say "  go <dir>:            move the player north, east, south or west[line break]";
	say "  examine ...:         examine something more closely[line break]";
	say "  eat ...:             eat edible food[line break]";
	say "  open ...:            open a door or a container[line break]";
	say "  close ...:           close a door or a container[line break]";
	say "  drop ...:            drop an object on the floor[line break]";
	say "  take ...:            take an object that is on the floor[line break]";
	say "  put ... on ...:      place an object on a supporter[line break]";
	say "  take ... from ...:   take an object from a container or a supporter[line break]";
	say "  insert ... into ...: place an object into a container[line break]";
	say "  lock ... with ...:   lock a door or a container with a key[line break]";
	say "  unlock ... with ...: unlock a door or a container with a key[line break]";

Understand "help" as displaying help message.

Taking all is an action applying to nothing.
Check taking all:
	say "You have to be more specific!";
	rule fails.

Understand "take all" as taking all.
Understand "get all" as taking all.
Understand "pick up all" as taking all.

Understand "take each" as taking all.
Understand "get each" as taking all.
Understand "pick up each" as taking all.

Understand "take everything" as taking all.
Understand "get everything" as taking all.
Understand "pick up everything" as taking all.

The extra description command option is a truth state that varies.
The extra description command option is usually false.

Turning on the extra description command option is an action applying to nothing.
Carry out turning on the extra description command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra description command option is true.

Understand "tw-extra-infos description" as turning on the extra description command option.

The extra inventory command option is a truth state that varies.
The extra inventory command option is usually false.

Turning on the extra inventory command option is an action applying to nothing.
Carry out turning on the extra inventory command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra inventory command option is true.

Understand "tw-extra-infos inventory" as turning on the extra inventory command option.

The extra score command option is a truth state that varies.
The extra score command option is usually false.

Turning on the extra score command option is an action applying to nothing.
Carry out turning on the extra score command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra score command option is true.

Understand "tw-extra-infos score" as turning on the extra score command option.

The extra moves command option is a truth state that varies.
The extra moves command option is usually false.

Turning on the extra moves command option is an action applying to nothing.
Carry out turning on the extra moves command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra moves command option is true.

Understand "tw-extra-infos moves" as turning on the extra moves command option.

To trace the actions:
	(- trace_actions = 1; -).

Tracing the actions is an action applying to nothing.
Carry out tracing the actions:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	trace the actions;

Understand "tw-trace-actions" as tracing the actions.

The restrict commands option is a truth state that varies.
The restrict commands option is usually false.

Turning on the restrict commands option is an action applying to nothing.
Carry out turning on the restrict commands option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the restrict commands option is true.

Understand "restrict commands" as turning on the restrict commands option.

The taking allowed flag is a truth state that varies.
The taking allowed flag is usually false.

Before removing something from something:
	now the taking allowed flag is true.

After removing something from something:
	now the taking allowed flag is false.

Before taking a thing (called the object) when the object is on a supporter (called the supporter):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [supporter] instead.";
		rule fails.

Before of taking a thing (called the object) when the object is in a container (called the container):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [container] instead.";
		rule fails.

Understand "take [something]" as removing it from.

Rule for supplying a missing second noun while removing:
	if restrict commands option is false and noun is on a supporter (called the supporter):
		now the second noun is the supporter;
	else if restrict commands option is false and noun is in a container (called the container):
		now the second noun is the container;
	else:
		try taking the noun;
		say ""; [Needed to avoid printing a default message.]

The version number is always 1.

Reporting the version number is an action applying to nothing.
Carry out reporting the version number:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	say "[version number]".

Understand "tw-print version" as reporting the version number.

Reporting max score is an action applying to nothing.
Carry out reporting max score:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	if maximum score is -32768:
		say "infinity";
	else:
		say "[maximum score]".

Understand "tw-print max_score" as reporting max score.

To print id of (something - thing):
	(- print {something}, "^"; -).

Printing the id of player is an action applying to nothing.
Carry out printing the id of player:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of player.

Printing the id of EndOfObject is an action applying to nothing.
Carry out printing the id of EndOfObject:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of EndOfObject.

Understand "tw-print player id" as printing the id of player.
Understand "tw-print EndOfObject id" as printing the id of EndOfObject.

There is a EndOfObject.

