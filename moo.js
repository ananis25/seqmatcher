function rulesWidget(initialRules) {
  // set up initial rules
  let rules = initialRules ? JSON.parse(JSON.stringify(initialRules)) : [];

  // set up html
  const html = d3.create("html");
  const element = html.node();
  element.value = JSON.parse(JSON.stringify(rules));
  element.dispatchEvent(new CustomEvent("input"));

  // create title row
  const titleRow = html.append("div").style("width", "100%");
  titleRow
    .append("div")
    .style("display", "inline-block")
    .style("width", "250px")
    .style("font", "700 0.9rem sans-serif")
    .html("<b>Rules to filter & modify sequences:</b>");
  const addRuleButton = titleRow
    .append("button")
    .attr("class", "add-rule")
    .style("display", "inline-block")
    .style("width", "100px")
    .text("Add Rule");
  const submitRulesButton = titleRow
    .append("button")
    .attr("class", "submit-rules")
    .style("display", "inline-block")
    .style("width", "100px")
    .style("margin", "10px 0px 10px 10px")
    .attr("disabled", true)
    .text("Submit Rules");
  titleRow
    .append("hr")
    .attr("class", "separator")
    .style("margin", "-10px 0px 0px 0px");

  // display input areas for rules
  const rulesInputArea = html.append("div");
  displayInputRules();

  // handle "add rule" button click
  html.select(".add-rule").on("click", () => {
    rules.push({
      id: rules.length > 0 ? Math.max(...rules.map((r) => r.id)) + 1 : 1,
      matchType: "_REGULAR",
      matchPatterns: [""],
      replacePattern: null,
    });
    displayInputRules();
  });

  // handle "submit rules" button click
  html.select(".submit-rules").on("click", () => {
    submitRulesButton.attr("disabled", true);
    rules = rules.filter((r) => !!r.matchPatterns[0]);
    rules.forEach((r) => {
      if (r.matchType == "_REPLACE" && !r.replacePattern) {
        r.matchType = "_REGULAR";
        r.replacePattern = null;
      }
      if (r.matchPatterns.length == 3 && !r.matchPatterns[2]) {
        r.matchPatterns.pop();
      }
      if (r.matchPatterns.length == 2 && !r.matchPatterns[1]) {
        r.matchPatterns.pop();
        r.matchType = "_REGULAR";
      }
    });
    element.value = JSON.parse(JSON.stringify(rules));
    element.dispatchEvent(new CustomEvent("input"));
    displayInputRules();
  });

  // add input areas for rules
  function displayInputRules() {
    html
      .select(".separator")
      .style("display", rules.length > 0 ? "block" : "none");

    let ruleInput = rulesInputArea
      .selectAll("div")
      .data(rules, (d) => d.id)
      .join("div")
      .style("display", "grid")
      .style("grid-template-columns", "repeat(18, 1fr)")
      .style("grid-row-gap", "5px")
      .style("width", "100%");

    // first row with match pattern
    ruleInput
      .append("div")
      .style("grid-column", 1)
      .style("grid-row", 1)
      .style("text-align", "center")
      .text((d, i) => "(" + (i + 1) + ")");
    ruleInput
      .append("div")
      .style("grid-column", 2)
      .style("grid-row", 1)
      .style("text-align", "center")
      .style("display", (d, i) =>
        rules[i].matchType == "_NOT" ? "block" : "none"
      )
      .text("NOT");
    ruleInput
      .append("div")
      .style("grid-column", "3 / 18")
      .style("grid-row", 1)
      .append("textarea")
      .style("width", "100%")
      .attr("rows", 1)
      .attr("class", "match-pattern")
      .attr("placeholder", "match pattern")
      .datum((d) => d)
      .html((d, i) => rules[i].matchPatterns[0])
      .on("input change", () => {
        d3.event.stopPropagation();
        d3.event.preventDefault();
        submitRulesButton.attr("disabled", null);
      })
      .on("blur", (d, i, nodes) => {
        rules[i].matchPatterns[0] = nodes[i].value;
      });
    ruleInput
      .append("div")
      .style("grid-column", "19")
      .style("grid-row", 1)
      .append("button")
      .attr("class", "delete-rule")
      .datum((d) => d)
      .text("X")
      .on("click", (d) => {
        submitRulesButton.attr("disabled", null);
        rules = rules.filter((r) => r.id != d.id);
        displayInputRules();
      });

    // rows of OR inputs
    ruleInput
      .append("div")
      .style("grid-column", 2)
      .style("grid-row", 2)
      .style("text-align", "center")
      .style("display", (d, i) =>
        rules[i].matchPatterns.length > 1 ? "block" : "none"
      )
      .text("OR");
    ruleInput
      .append("div")
      .style("grid-column", "3 / 18")
      .style("grid-row", 2)
      .style("display", (d, i) =>
        rules[i].matchPatterns.length > 1 ? "block" : "none"
      )
      .append("textarea")
      .style("width", "100%")
      .attr("rows", 1)
      .attr("class", "match-pattern-2")
      .attr("placeholder", "match pattern 2")
      .datum((d) => d)
      .html((d, i) => rules[i].matchPatterns[1])
      .on("input change", () => {
        d3.event.stopPropagation();
        d3.event.preventDefault();
        submitRulesButton.attr("disabled", null);
      })
      .on("blur", (d, i, nodes) => {
        rules[i].matchPatterns[1] = nodes[i].value;
      });
    ruleInput
      .append("div")
      .style("grid-column", 2)
      .style("grid-row", 3)
      .style("text-align", "center")
      .style("display", (d, i) =>
        rules[i].matchPatterns.length > 2 ? "block" : "none"
      )
      .text("OR");
    ruleInput
      .append("div")
      .style("grid-column", "3 / 18")
      .style("grid-row", 3)
      .style("display", (d, i) =>
        rules[i].matchPatterns.length > 2 ? "block" : "none"
      )
      .append("textarea")
      .style("width", "100%")
      .attr("rows", 1)
      .attr("class", "match-pattern-3")
      .attr("placeholder", "match pattern 3")
      .datum((d) => d)
      .html((d, i) => rules[i].matchPatterns[2])
      .on("input change", () => {
        d3.event.stopPropagation();
        d3.event.preventDefault();
        submitRulesButton.attr("disabled", null);
      })
      .on("blur", (d, i, nodes) => {
        rules[i].matchPatterns[2] = nodes[i].value;
      });

    // row of buttons to add rule complexity (OR, NOT, REPLACE)
    ruleInput
      .append("div")
      .style("grid-column", 3)
      .style("grid-row", (d, i) => rules[i].matchPatterns.length + 1)
      .style("display", (d, i) =>
        ["_REGULAR", "_OR"].includes(rules[i].matchType) &&
        rules[i].matchPatterns.length < 3
          ? "block"
          : "none"
      )
      .append("button")
      .style("font", "0.75em Arial")
      .text("or")
      .on("click", (d, i) => {
        rules[i].matchType = "_OR";
        rules[i].matchPatterns.push("");
        displayInputRules();
      });
    ruleInput
      .append("div")
      .style("grid-column", 4)
      .style("grid-row", (d, i) => rules[i].matchPatterns.length + 1)
      .style("display", (d, i) =>
        rules[i].matchType == "_REGULAR" ? "block" : "none"
      )
      .append("button")
      .style("font", "0.75em Arial")
      .text("not")
      .on("click", (d, i) => {
        rules[i].matchType = "_NOT";
        displayInputRules();
      });
    ruleInput
      .append("div")
      .style("grid-column", 5)
      .style("grid-row", (d, i) => rules[i].matchPatterns.length + 1)
      .style("display", (d, i) =>
        rules[i].matchType == "_REGULAR" ? "block" : "none"
      )
      .append("button")
      .style("font", "0.75em Arial")
      .text("replace")
      .on("click", (d, i) => {
        rules[i].matchType = "_REPLACE";
        displayInputRules();
      });

    // replace pattern row
    ruleInput
      .append("div")
      .style("grid-column", "2")
      .style("grid-row", (d, i) => rules[i].matchPatterns.length + 2)
      .style("text-align", "center")
      .style("display", (d, i) =>
        rules[i].matchType == "_REPLACE" ? "block" : "none"
      )
      .html("&#8627;");
    ruleInput
      .append("div")
      .style("grid-column", "3 / 18")
      .style("grid-row", (d, i) => rules[i].matchPatterns.length + 2)
      .style("display", (d, i) =>
        rules[i].matchType == "_REPLACE" ? "block" : "none"
      )
      .append("textarea")
      .style("width", "100%")
      .attr("rows", 1)
      .attr("class", "replace-pattern")
      .attr("placeholder", "new pattern")
      .datum((d) => d)
      .html((d, i) => rules[i].replacePattern)
      .on("input change", () => {
        d3.event.stopPropagation();
        d3.event.preventDefault();
        submitRulesButton.attr("disabled", null);
      })
      .on("blur", (d, i, nodes) => {
        rules[i].replacePattern = nodes[i].value;
      });

    // separator
    ruleInput
      .append("div")
      .style("grid-column", "1 / 19")
      .style("grid-row", (d, i) => rules[i].matchPatterns.length + 3)
      .append("hr")
      .style("margin", "-10px 0px 0px 0px");
  }

  return element;
}

// executeRules(sequences, rules) - executes filter and replace rules on provided sequences in specificed order
const executeRules = function (sequences, rules) {
  // if no rules, return sequences
  if (rules.length == 0) {
    return sequences;
  }

  // if provided string rules, convert them to object rules
  const objRules =
    typeof rules[0].matchPatterns[0] === "string" ? readRules(rules) : rules;

  // apply rules one by one
  let currSequences = sequences;
  let newSequences,
    matchedSequences,
    allowOverlaps,
    matchedRangeIdx,
    mtchPtrn,
    areValidMatches,
    isValidMatch;
  objRules.forEach((rule) => {
    // match sequences based on provided patterns
    matchedSequences = [];
    for (let i = 0; i < rule.matchPatterns.length; i++) {
      matchedSequences.push(matchPattern(currSequences, rule.matchPatterns[i]));
    }

    // apply NOT/OR/REPLACE logic
    if (rule.matchType == "_NOT") {
      // filter to sequences without matches
      newSequences = matchedSequences[0]
        .filter((seq) => seq.matchedPattern.matchedEventsIdx.length == 0)
        .map((seq) => {
          delete seq.matchedPattern;
          return seq;
        });
    } else if (rule.matchType == "_REPLACE") {
      // replace matched sequences
      newSequences = replacePattern(matchedSequences[0], rule.replacePattern);
    } else if (rule.matchType == "_REGULAR") {
      // filter sequences based on matches
      newSequences = extractPattern(matchedSequences[0]);
    } else {
      // OR rule
      // filter sequences based on matches for several patterns with deduping
      allowOverlaps = rule.matchPatterns.some((ptrn) => ptrn.allowOverlaps);
      newSequences = extractPattern(matchedSequences[0]);
      matchedRangeIdx = matchedSequences[0].map(
        (seq) => seq.matchedPattern.matchedRangeIdx
      );
      for (let i = 1; i < rule.matchPatterns.length; i++) {
        for (let j = 0; j < matchedSequences[i].length; j++) {
          mtchPtrn = matchedSequences[i][j].matchedPattern;
          // find valid matches
          areValidMatches = [];
          for (let k = 0; k < mtchPtrn.matchedRangeIdx.length; k++) {
            isValidMatch = true;
            for (let l = 0; l < matchedRangeIdx[j].length; l++) {
              if (
                (!allowOverlaps &&
                  !(
                    mtchPtrn.matchedRangeIdx[k][0] > matchedRangeIdx[j][l][1]
                  ) &&
                  !(
                    mtchPtrn.matchedRangeIdx[k][1] < matchedRangeIdx[j][l][0]
                  )) || // overlapping
                (mtchPtrn.matchedRangeIdx[k][0] == matchedRangeIdx[j][l][0] &&
                  mtchPtrn.matchedRangeIdx[k][1] == matchedRangeIdx[j][l][1])
              ) {
                // duplicate
                isValidMatch = false;
                break;
              }
            }
            areValidMatches.push(isValidMatch);
          }
          // remove overlapping or duplicate matches
          mtchPtrn.matchedEventsIdx = mtchPtrn.matchedEventsIdx.filter(
            (x, idx) => areValidMatches[idx]
          );
          mtchPtrn.matchedEventsCount = mtchPtrn.matchedEventsCount.filter(
            (x, idx) => areValidMatches[idx]
          );
          mtchPtrn.matchedRangeIdx = mtchPtrn.matchedRangeIdx.filter(
            (x, idx) => areValidMatches[idx]
          );
        }
        newSequences = newSequences.concat(extractPattern(matchedSequences[i]));
        matchedSequences[i].forEach((seq, l) => {
          matchedRangeIdx[l] = matchedRangeIdx[l].concat(
            seq.matchedPattern.matchedRangeIdx
          );
        });
      }
    }
    currSequences = JSON.parse(JSON.stringify(newSequences));
  });

  return currSequences;
};

// replacePattern(sequences, replacePattern, [matchPattern]) function returns a copy of the "sequences" with events/sub-sequences matching a pattern replaced by events/sub-sequences from the "replacePattern". "matchPattern" input is optional; if not provided, "sequences" needs to have "matchedPattern" property for each sequence.

const replacePattern = function (sequences, replacePattern, matchPattern) {
  // handle bad input
  if (
    !sequences ||
    !replacePattern ||
    !Array.isArray(sequences) ||
    (!sequences[0].hasOwnProperty("matchedPattern") && !matchPattern)
  ) {
    return null;
  }

  // if provided with match pattern input, match it first
  if (matchPattern) {
    sequences = matchPattern(sequences, matchPattern);
  }
  const mPtrn = sequences[0].matchedPattern.pattern;
  // can't replace if matches overlap
  if (mPtrn.allowOverlaps) {
    return null;
  }

  // read replace pattern, if needed
  const rPtrn =
    typeof replacePattern === "string"
      ? readPattern(replacePattern)
      : replacePattern;
  // can't replace if replace pattern is unreadable
  if (!rPtrn) {
    return null;
  }

  // replace matched sequences one-by-one
  let replacedSequences = [];
  let mtchPtrn, events, pos, ev, nEv, nSeq;
  for (let seq of sequences) {
    // there could be more than 1 match per sequence, so need to iterate over all matches
    events = [];
    pos = 0;
    for (let i = 0; i < seq.matchedPattern.matchedEventsIdx.length; i++) {
      mtchPtrn = seq.matchedPattern;

      // copy seq events before the match starts
      for (let j = pos; j < mtchPtrn.matchedRangeIdx[i][0]; j++) {
        events.push(Object.assign({}, seq.events[j]));
      }

      // extract matched events with custom names
      ev = {};
      for (let eventName in mtchPtrn.matchedEventsIdx[i]) {
        ev[eventName] = seq.events.slice(
          mtchPtrn.matchedEventsIdx[i][eventName],
          mtchPtrn.matchedEventsIdx[i][eventName] +
            mtchPtrn.matchedEventsCount[i][eventName]
        );
        // ev[eventName] = seq.events[mtchPtrn.matchedEventsIdx[i][eventName] + mtchPtrn.matchedEventsCount[i][eventName] - 1];
      }

      // assemble new events based on the match and the replace pattern and add them
      for (let evPtrn of rPtrn.events) {
        if (evPtrn.customName && evPtrn.customName.startsWith("ALL@")) {
          // copy over list of matched events by custom name
          events = events.concat(ev[evPtrn.customName.slice(4)]);
        } else if (
          evPtrn.customName &&
          evPtrn.customName.startsWith("REVERSE@")
        ) {
          // copy over a reversed list of matched events by custom name
          events = events.concat(ev[evPtrn.customName.slice(8)].reverse());
        } else {
          nEv = {};
          // add event name
          if (evPtrn.includeNames.length > 0) {
            if (evPtrn.includeNames[0][0] == "@") {
              nEv._eventName =
                ev[evPtrn.includeNames[0].slice(1)][
                  ev[evPtrn.includeNames[0].slice(1)].length - 1
                ]._eventName;
            } else {
              nEv._eventName = evPtrn.includeNames[0];
            }
          } else {
            nEv._eventName = "NO_NAME";
          }
          // add event properties
          for (let prop of evPtrn.properties) {
            if (prop.name[0] == "@") {
              // copy properties from custom name event
              let customEv =
                ev[prop.name.slice(1)][ev[prop.name.slice(1)].length - 1];
              for (let customProp in customEv) {
                if (customProp != "_eventName") {
                  nEv[customProp] = customEv[customProp];
                }
              }
            } else if (prop.value[0] == "@") {
              nEv[prop.name] =
                ev[prop.value.slice(1)][ev[prop.value.slice(1)].length - 1][
                  prop.name
                ];
            } else {
              nEv[prop.name] = prop.value;
            }
          }
          // execute javascript code
          evPtrn.jsCode.forEach((js) => {
            eval(js);
          });
          if (nEv._eventName == null) {
            nEv._eventName = "NO_NAME";
          } else {
            nEv._eventName = String(nEv._eventName);
          }
          events = events.concat(
            new Array(evPtrn.minCount ? evPtrn.minCount : 1).fill(nEv)
          );
        }
      }

      pos = mtchPtrn.matchedRangeIdx[i][1] + 1;
    }

    // add remaining seq events after the end of the last match
    for (let j = pos; j < seq.events.length; j++) {
      events.push(Object.assign({}, seq.events[j]));
    }

    // construct full sequence
    nSeq = {};
    for (let prop in seq) {
      if (!["events", "matchedPattern"].includes(prop)) {
        nSeq[prop] = seq[prop];
      }
    }
    nSeq.events = events;
    for (let prop of rPtrn.sequenceProperties) {
      nSeq[prop.name] = prop.value;
    }

    // execute sequence-level javascript code
    rPtrn.jsCode.forEach((js) => {
      eval(js);
    });

    replacedSequences.push(nSeq);
  }

  return replacedSequences;
};

// extractPattern(sequences, [pattern]) function returns sub-sequences extracted from "sequences" by matching a pattern. "pattern" input is optional; if not provided, "sequences" needs to have "matchedPattern" property for each sequence.

const extractPattern = function (sequences, pattern) {
  // handle bad input
  if (
    !sequences ||
    !Array.isArray(sequences) ||
    (!sequences[0].hasOwnProperty("matchedPattern") && !pattern)
  ) {
    return null;
  }

  // if provided with pattern input, match it first
  if (pattern) {
    sequences = matchPattern(sequences, pattern);
  }

  // extract matched sequences one-by-one
  let extractedSequences = [];
  let newSequence;
  for (let seq of sequences) {
    // there could be more than 1 match per sequence, so need to iterate over all matches
    for (let i = 0; i < seq.matchedPattern.matchedRangeIdx.length; i++) {
      newSequence = {};
      for (let prop in seq) {
        if (!["events", "matchedPattern"].includes(prop)) {
          newSequence[prop] = seq[prop];
        }
      }
      newSequence.events = seq.events.slice(
        seq.matchedPattern.matchedRangeIdx[i][0],
        seq.matchedPattern.matchedRangeIdx[i][1] + 1
      );
      extractedSequences.push(newSequence);
    }
  }

  return extractedSequences;
};

// matchPattern(sequences, pattern) function returns the original array of "sequences" with an added sequence-level property "matchedPattern", based on matching the provided "pattern" (object or string) to each sequence in the array. The new sequence-level property consists of:
// - pattern: input pattern in the object form
// - matchedEventsIdx: array of objects of indices of events in the original sequence matched to pattern event customNames and also "_N" strings, where N = index of an event in the pattern object (array length is 1 if pattern.whereToMatch = "_FIRST")
// - matchedEventsCount: array of objects of counts of events in the original sequence matched to pattern event customNames and also "_N" strings, where N = index of an event in the pattern object (array length is 1 if pattern.whereToMatch = "_FIRST")
// - matchedRangeIdx: array of pairs [start_event_index, end_event_index] for matches to the pattern (array length is 1 if pattern.whereToMatch = "_FIRST")

const matchPattern = function (sequences, pattern) {
  // handle bad input
  if (
    !sequences ||
    !pattern ||
    !["string", "object"].includes(typeof pattern) ||
    (typeof pattern === "object" &&
      (!pattern.hasOwnProperty("events") || pattern.events.length == 0))
  ) {
    return null;
  }

  // if needed, convert pattern to a pattern object
  const patternObj =
    typeof pattern === "string" ? readPattern(pattern) : pattern;

  // match sequences to the pattern one-by-one
  let matchedSequences = JSON.parse(JSON.stringify(sequences));
  let pos,
    matchIdx,
    matchCount,
    ev,
    matchedEventsIdx,
    matchedEventsCount,
    hasMatchedSeqProperties;
  for (let seq of matchedSequences) {
    // seed matchedPattern property
    seq.matchedPattern = {
      pattern: patternObj,
      matchedEventsIdx: [],
      matchedEventsCount: [],
      matchedRangeIdx: [],
    };

    // match sequence properties
    hasMatchedSeqProperties = true;
    for (let prop of patternObj.sequenceProperties) {
      if (Array.isArray(prop.value)) {
        // special handling of matching to a list of possible values
        let includes =
          seq.hasOwnProperty(prop.name) && prop.value.includes(seq[prop.name]);
        if (prop.operator == "==" ? !includes : includes) {
          hasMatchedSeqProperties = false;
        }
      } else {
        if (
          (seq.hasOwnProperty(prop.name) &&
            !eval("seq[prop.name]" + prop.operator + "prop.value")) ||
          (!seq.hasOwnProperty(prop.name) && prop.operator != "!=")
        ) {
          hasMatchedSeqProperties = false;
        }
      }
    }
    if (!hasMatchedSeqProperties) {
      continue;
    }

    // match the pattern
    pos = 0;
    while (pos < (patternObj.matchSequenceStart ? 1 : seq.events.length)) {
      matchIdx = [];
      matchCount = [];
      ev = {};
      matchedEventsIdx = {};
      matchedEventsCount = {};
      if (hasMatchedSequence(seq, pos, 0, matchIdx, matchCount, ev)) {
        // recursive matching
        // matched sequence to the pattern - create matchedEventsIdx
        for (let i = 0; i < matchIdx.length; i++) {
          if (patternObj.events[i].customName) {
            matchedEventsIdx[patternObj.events[i].customName.slice(1)] =
              matchIdx[i]; // remove '@' in custom names
            matchedEventsCount[patternObj.events[i].customName.slice(1)] =
              matchCount[i]; // remove '@' in custom names
          }
          matchedEventsIdx["_" + i] = matchIdx[i];
          matchedEventsCount["_" + i] = matchCount[i];
        }

        // match javascript conditions
        ev = {};
        for (let eventName in matchedEventsIdx) {
          ev[eventName] = seq.events.slice(
            matchedEventsIdx[eventName],
            matchedEventsIdx[eventName] + matchedEventsCount[eventName]
          );
        }
        if (patternObj.jsCode.every((cond) => eval(cond))) {
          // remove '@' in custom names
          // passed javascript conditions - add the match to the seq object
          seq.matchedPattern.matchedEventsIdx.push(matchedEventsIdx);
          seq.matchedPattern.matchedEventsCount.push(matchedEventsCount);
          seq.matchedPattern.matchedRangeIdx.push([
            patternObj.startEventIdx != null
              ? matchIdx[patternObj.startEventIdx]
              : 0,
            patternObj.endEventIdx != null
              ? matchIdx[patternObj.endEventIdx] +
                matchCount[patternObj.endEventIdx] -
                1
              : seq.events.length - 1,
          ]);
          if (patternObj.whereToMatch == "_FIRST") {
            // requested only 1 match and found it - no need to match more
            break;
          } else {
            pos = patternObj.allowOverlaps
              ? matchIdx[0] + 1
              : seq.matchedPattern.matchedRangeIdx[
                  seq.matchedPattern.matchedRangeIdx.length - 1
                ][1] + 1;
          }
        } else {
          // didn't pass javascript conditions - keep matching
          pos = matchIdx[0] + 1;
        }
      } else {
        // didn't find a match starting at current position, try next one
        pos += 1;
      }
    }
  }

  return matchedSequences;

  // iterative function to match events to a pattern
  function hasMatchedSequence(
    seq,
    startSeqPos,
    startPatternIdx,
    matchIdx,
    matchCount,
    ev
  ) {
    const evP = patternObj.events[startPatternIdx];
    let pos1;

    // special handling of 0 event count
    if (evP.maxCount == 0) {
      if (hasMatchedEvent(seq, startSeqPos, evP, ev)) {
        return false;
      }
    }

    // match events minimum number of times specified in the pattern
    for (pos1 = startSeqPos; pos1 < startSeqPos + evP.minCount; pos1++) {
      if (pos1 == seq.events.length || !hasMatchedEvent(seq, pos1, evP, ev)) {
        // couldn't match minimum number of times specified in the pattern
        return false;
      }
    }
    matchIdx.push(startSeqPos);
    matchCount.push(evP.minCount);
    if (evP.customName) {
      ev[evP.customName.slice(1)] = seq.events[startSeqPos + evP.minCount - 1];
    }

    let maxPos =
      evP.maxCount == null
        ? seq.events.length - 1
        : Math.min(startSeqPos + evP.maxCount - 1, seq.events.length - 1);
    if (startPatternIdx < patternObj.events.length - 1) {
      // pattern has more events to match

      // match events until the maximum number of times specified in the pattern
      for (pos1 = startSeqPos + evP.minCount; pos1 <= maxPos; pos1++) {
        // try to match the next event in the pattern
        if (
          hasMatchedSequence(
            seq,
            pos1,
            startPatternIdx + 1,
            matchIdx,
            matchCount,
            ev
          )
        ) {
          // matched the rest of the sequence
          return true;
        }
        // try to match the current event in the pattern
        if (!hasMatchedEvent(seq, pos1, evP, ev)) {
          // couldn't further match current event to support more tries for the consecutive events
          matchIdx.pop();
          matchCount.pop();
          if (evP.customName) {
            delete ev[evP.customName.slice(1)];
          }
          return false;
        } else {
          matchCount[matchCount.length - 1] += 1;
          if (evP.customName) {
            ev[evP.customName.slice(1)] = seq.events[pos1];
          }
        }
      }

      // match the next event to the next event in the pattern
      if (maxPos < seq.events.length - 1) {
        // sequence has more events to match to
        if (
          hasMatchedSequence(
            seq,
            maxPos + 1,
            startPatternIdx + 1,
            matchIdx,
            matchCount,
            ev
          )
        ) {
          // matched the rest of the sequence
          return true;
        } else {
          // couldn't match next event in all possible positions
          matchIdx.pop();
          matchCount.pop();
          if (evP.customName) {
            delete ev[evP.customName.slice(1)];
          }
          return false;
        }
      } else {
        // reached the end of the sequence but there are more events in the pattern
        matchIdx.pop();
        matchCount.pop();
        if (evP.customName) {
          delete ev[evP.customName.slice(1)];
        }
        return false;
      }
    } else {
      // it is the last event in the pattern

      // match more of the current event if haven't reached maxCount
      for (pos1 = startSeqPos + evP.minCount; pos1 <= maxPos; pos1++) {
        if (!hasMatchedEvent(seq, pos1, evP, ev)) {
          break;
        }
      }
      if (pos1 > startSeqPos + evP.minCount) {
        matchCount[matchCount.length - 1] += pos1 - startSeqPos - evP.minCount;
        if (evP.customName) {
          ev[evP.customName.slice(1)] = seq.events[pos1 - 1];
        }
      }

      // check the condition of matching the end of the sequence
      if (patternObj.matchSequenceEnd && pos1 < seq.events.length) {
        matchIdx.pop();
        matchCount.pop();
        if (evP.customName) {
          delete ev[evP.customName.slice(1)];
        }
        return false;
      } else {
        return true;
      }
    }
  }

  // checking whether a specific event in a sequence matches a specific event in a pattern
  function hasMatchedEvent(seq, pos, evP, ev) {
    const currEv = seq.events[pos];
    let matchedRefEvWithoutProp, includes;
    let hasMatched = true;

    // check includeNames pattern
    if (
      hasMatched &&
      evP.includeNames.length > 0 &&
      evP.includeNames.every((name) =>
        name[0] == "@"
          ? currEv._eventName != ev[name.slice(1)]._eventName
          : currEv._eventName != name
      )
    ) {
      hasMatched = false;
    }

    // check excludeNames pattern
    if (
      hasMatched &&
      evP.excludeNames.length > 0 &&
      evP.excludeNames.some((name) =>
        name[0] == "@"
          ? currEv._eventName == ev[name.slice(1)]._eventName
          : currEv._eventName == name
      )
    ) {
      hasMatched = false;
    }

    // check properties pattern
    if (hasMatched && evP.properties.length > 0) {
      for (let prop of evP.properties) {
        if (!currEv.hasOwnProperty(prop.name)) {
          // special handling of sequence event not having the matching property
          matchedRefEvWithoutProp =
            (prop.operator == "==" &&
              (Array.isArray(prop.value) ? prop.value : [prop.value]).some(
                (v) => v[0] == "@" && !ev[v.slice(1)].hasOwnProperty(prop.name)
              )) ||
            (prop.operator == "!=" &&
              !(Array.isArray(prop.value) ? prop.value : [prop.value]).some(
                (v) => v[0] == "@" && !ev[v.slice(1)].hasOwnProperty(prop.name)
              ));
          if (!matchedRefEvWithoutProp) {
            hasMatched = false;
          }
        } else {
          if (Array.isArray(prop.value)) {
            // special handling of matching to a list of possible values
            includes = prop.value.some((v) =>
              v[0] == "@"
                ? ev[v.slice(1)].hasOwnProperty(prop.name) &&
                  currEv[prop.name] == ev[v.slice(1)][prop.name]
                : currEv[prop.name] == v
            );
            if (prop.operator == "==" ? !includes : includes) {
              hasMatched = false;
            }
          } else if (prop.value[0] == "@") {
            // value is referencing a custom event name
            if (
              ev.hasOwnProperty(prop.value.slice(1))
                ? !eval(
                    "currEv[prop.name]" +
                      prop.operator +
                      "ev[prop.value.slice(1)][prop.name]"
                  )
                : prop.operator != "!="
            ) {
              hasMatched = false;
            }
          } else {
            // provided direct value
            if (!eval("currEv[prop.name]" + prop.operatohar + "prop.value")) {
              hasMatched = false;
            }
          }
        }
      }
    }

    return hasMatched;
  }
};

// readRules(stringRules) - parses a list of interaction rules with well-formed strings for matching and replacing patterns into a pattern object, which can be used with the executeRules() function
const readRules = function (stringRules) {
  // parse rule-by-rule
  let objectRules = stringRules.map((strRule) => {
    let objRule = JSON.parse(JSON.stringify(strRule));

    // parse all match rules
    objRule.matchPatterns = strRule.matchPatterns.map((pattern) =>
      readPattern(pattern)
    );
    // if there is a replace pattern, parse it too
    if (strRule.matchType == "_REPLACE") {
      objRule.replacePattern = readPattern(strRule.replacePattern);
    }
    return objRule;
  });

  return objectRules;
};

// readPattern(patternString) function parses a well-formed string into a pattern object, which can be used with the matchPattern() function
const readPattern = function (patternString) {
  const string = patternString.split(" ").join(""); // remove all spaces to simplify parsing
  let pattern, pos;

  // seed the pattern object
  pattern = {
    whereToMatch: "_ALL",
    allowOverlaps: false,
    sequenceProperties: [],
    jsCode: [],
    matchSequenceStart: false,
    matchSequenceEnd: false,
    startEventIdx: null,
    endEventIdx: null,
    events: [],
  };
  let foundStartSymbol = false;
  let foundEndSymbol = false;

  // read start condition
  if (string.startsWith("|-")) {
    pattern.matchSequenceStart = true;
    pos = 2;
  } else if (string.startsWith("^-")) {
    pattern.matchSequenceStart = false;
    foundStartSymbol = true;
    pos = 2;
  } else if (string.startsWith("-")) {
    pattern.matchSequenceStart = false;
    pos = 1;
  } else if (string.startsWith("(") || string.startsWith("[")) {
    pattern.matchSequenceStart = false;
    pattern.startEventIdx = 0; // start subsequence with the first pattern event
    pos = 0;
  } else {
    console.log("start condition error - " + pos);
    return null;
  }

  // read all events and distances
  let startMetadataPos =
    string.indexOf("{{") >= 0 ? string.indexOf("{{") : string.length;
  let lastEventPos = Math.max(
    string.slice(0, startMetadataPos).lastIndexOf(")"),
    string.slice(0, startMetadataPos).lastIndexOf("]")
  );
  let event;
  while (pos <= lastEventPos) {
    event = {
      customName: null,
      minCount: null,
      maxCount: null,
      includeNames: [],
      excludeNames: [],
      jsCode: [],
      properties: [],
    };

    // skip over links
    if (string[pos] != "-" && pattern.events.length > 0) {
      console.log("link between events error - " + pos);
      return null;
    }
    while (string[pos] == "-") {
      pos += 1;
    }

    // read event
    event = readEvent(event);
    if (event == null) {
      return null;
    } else {
      pattern.events.push(event);
    }
  }

  // read end condition
  if (string.slice(pos).startsWith("-|")) {
    pattern.matchSequenceEnd = true;
    pos += 2;
  } else if (string.slice(pos).startsWith("-$") && !foundEndSymbol) {
    pattern.matchSequenceEnd = false;
    foundEndSymbol = true;
    pos += 2;
  } else if (string.slice(pos).startsWith("-")) {
    pattern.matchSequenceEnd = false;
    pos += 1;
  } else if (string.slice(pos).startsWith("{{") || pos == string.length) {
    pattern.matchSequenceEnd = false;
    if (!foundEndSymbol) {
      pattern.endEventIdx = pattern.events.length - 1; // end subsequence with the last pattern event
    }
  } else {
    console.log("end condition error - " + pos);
    return null;
  }

  // read optional meta-parameters, sequence-level properties and javascript code
  if (pos == string.length) {
    return pattern;
  } else if (!string.slice(pos).startsWith("{{")) {
    console.log("start meta-parameters error - " + pos);
    return null;
  }
  let properties = readProperties("{{", "}}");
  if (properties == null) {
    return null;
  }
  for (let prop of properties) {
    if (prop.hasOwnProperty("jsCode")) {
      // javascript code - add to pattern
      pattern.jsCode.push(prop.jsCode);
    } else if (prop.name[0] == "_") {
      // meta-parameter - add to pattern
      pattern[prop.name.slice(1)] = prop.value;
    } else {
      // sequence-level property - add to sequenceProperties
      pattern.sequenceProperties.push(prop);
    }
  }

  return pattern;

  function readEvent(event) {
    // read range of counts
    let hasCounts = false;
    if (string[pos] == "[") {
      pos += 1;
      // read optional minCount
      if (string.slice(pos).search(/^\d+/) >= 0) {
        event.minCount = string.slice(pos).match(/^\d+/)[0];
        pos += event.minCount.length;
        event.minCount = +event.minCount;
      }
      // read optional ".."
      let range = false;
      if (string.slice(pos).startsWith("..")) {
        range = true;
        pos += 2;
      }
      // read optional maxCount
      if (string.slice(pos).search(/^\d+/) >= 0) {
        event.maxCount = string.slice(pos).match(/^\d+/)[0];
        pos += event.maxCount.length;
        event.maxCount = +event.maxCount;
      }
      // read closing bracket
      if (string[pos] != "]") {
        console.log("counts end error - " + pos);
        return null;
      }
      pos += 1;
      if (event.minCount != null && !range && !event.maxCount) {
        event.maxCount = event.minCount;
      }
      if (event.minCount == null) {
        event.minCount = 0;
      }
      hasCounts = true;
    } else {
      event.minCount = 1;
      event.maxCount = 1;
    }
    // read opening bracket
    if (string[pos] != "(") {
      if (hasCounts) {
        return event;
      } else {
        console.log("event start error - " + pos);
        return null;
      }
    }
    pos += 1;
    // read optional start symbol
    if (string[pos] == "^") {
      if (!foundStartSymbol && !foundEndSymbol) {
        foundStartSymbol = true;
        pattern.startEventIdx = pattern.events.length;
        pos += 1;
      } else {
        console.log("start symbol error - " + pos);
        return null;
      }
    }
    // read optional end symbol
    if (string[pos] == "$") {
      if (foundStartSymbol && !foundEndSymbol) {
        foundEndSymbol = true;
        pattern.endEventIdx = pattern.events.length;
        pos += 1;
      } else {
        console.log("end symbol error - " + pos);
        return null;
      }
    }
    // read optional custom event name
    if (string.slice(pos).search(/^(ALL|REVERSE)?@\w+/) >= 0) {
      event.customName = string.slice(pos).match(/^(ALL|REVERSE)?@\w+/)[0];
      pos += event.customName.length;
    }
    // read optional "not" sign for event names exclusion
    let isExclude = false;
    if (string[pos] == "!") {
      isExclude = true;
      pos += 1;
    }
    // read optional event names
    let needName = false;
    if (string[pos] == ":") {
      needName = true;
      pos += 1;
      while (string.slice(pos).search(/^@?\w+/) >= 0) {
        needName = false;
        let name = string.slice(pos).match(/^@?\w+/)[0];
        if (isExclude) {
          event.excludeNames.push(name);
        } else {
          event.includeNames.push(name);
        }
        pos += name.length;
        if (string[pos] == "|") {
          needName = true;
          pos += 1;
          if (string[pos] == ":") {
            pos += 1;
          }
        }
      }
    } else if (isExclude || needName) {
      console.log("event names error - " + pos);
      return null;
    }
    // read optional property names
    let properties = readProperties("{", "}");
    if (properties == null) {
      return null;
    }
    for (let prop of properties) {
      if (prop.hasOwnProperty("jsCode")) {
        // event-level javascript code
        event.jsCode.push(prop.jsCode);
      } else {
        // event-level property
        event.properties.push(prop);
      }
    }
    // read closing bracket
    if (string[pos] != ")") {
      console.log("event end error - " + pos);
      return null;
    }
    pos += 1;
    return event;
  }

  function readProperties(startBracket, endBracket) {
    // TODO: add support for complex property values - arrays and objects
    let properties = [];
    let property, needValue;
    if (!string.slice(pos).startsWith(startBracket)) {
      return [];
    } else {
      pos += startBracket.length;
      while (pos < string.length && !string.slice(pos).startsWith(endBracket)) {
        property = {
          name: null,
          operator: null,
          value: [],
        };
        // read javascript code
        if (string[pos] == "#") {
          pos += 1;
          let jsCodeEndPos = pos + string.slice(pos).indexOf("#");
          if (jsCodeEndPos < pos) {
            console.log("js code end symbol error - " + pos);
            return null;
          } else {
            // replace "@custom" shorthands with proper event objects "ev.custom"
            property.jsCode = string
              .slice(pos, jsCodeEndPos)
              //.replaceAll(/@\w+(?![\w\[])/g, '$&[$&.length-1]')
              .replaceAll(/@\w+/g, "ev.$&")
              .replaceAll("@", "");
            pos += string.slice(pos, jsCodeEndPos).length + 1;
          }
        } else {
          // read property
          if (string[pos] == "@") {
            // name is a reference to another event via a custom name (no need for operator or value)
            if (string.slice(pos).search(/^@\w+/) >= 0) {
              property.name = string.slice(pos).match(/^@\w+/)[0];
              pos += property.name.length;
            }
          } else {
            // name is a direct string
            if (string.slice(pos).search(/^\w+/) >= 0) {
              property.name = string.slice(pos).match(/^\w+/)[0];
              pos += property.name.length;
            } else {
              console.log("prop name error - " + pos);
              return null;
            }
            // read operator
            if (["==", "!=", ">=", "<="].includes(string.slice(pos, pos + 2))) {
              property.operator = string.slice(pos, pos + 2);
              pos += 2;
            } else if (["=", ":", ">", "<"].includes(string[pos])) {
              property.operator = ["=", ":"].includes(string[pos])
                ? "=="
                : string[pos];
              pos += 1;
            } else {
              console.log("prop operator error - " + pos);
              return null;
            }
            // read value(s)
            needValue = true;
            let value;
            // direct value(s)
            while (string.slice(pos).search(/^(?:\d+|@?\w+)/) >= 0) {
              needValue = false;
              value = string.slice(pos).match(/^(?:\d+|@?\w+)/)[0];
              pos += value.length;
              if (value == "true") {
                value = true;
              } else if (value == "false") {
                value = false;
              } else if (!isNaN(+value)) {
                value = +value;
              }
              property.value.push(value);
              if (string[pos] == "|") {
                needValue = true;
                pos += 1;
              }
            }
            if (needValue) {
              console.log("prop value(s) error - " + pos);
              return null;
            }
            if (property.value.length == 1) {
              property.value = property.value[0];
            }
          }
        }
        properties.push(property);
        if (string[pos] == ",") {
          pos += 1;
        }
      }
      if (string.slice(pos).startsWith(endBracket)) {
        pos += endBracket.length;
      } else {
        console.log("prop end symbol error - " + pos);
        return null;
      }
    }
    return properties;
  }
};
