% linprog argument errors: messages and identifiers
o = optimoptions('linprog', 'Display', 'off');
try, linprog([1; 1], [1 1 1], 1, [], [], [], [], o); catch ME, id1 = ME.identifier; msg1 = ME.message; end
try, linprog([1; 1], [1 1; 1 0], 1, [], [], [], [], o); catch ME, id2 = ME.identifier; msg2 = ME.message; end
try, linprog([1; 1], [], [], [1 1 1], 1, [], [], o); catch ME, id3 = ME.identifier; msg3 = ME.message; end
try, linprog([1; 1], [], [], [1 1], [1; 2], [], [], o); catch ME, id4 = ME.identifier; msg4 = ME.message; end
try, linprog(); catch ME, id5 = ME.identifier; msg5 = ME.message; end
try, linprog('abc', [], []); catch ME, id6 = ME.identifier; msg6 = ME.message; end
try, linprog([1; 1], [], [], [], [], [], [], 5); catch ME, id7 = ME.identifier; msg7 = ME.message; end
clear o ME
